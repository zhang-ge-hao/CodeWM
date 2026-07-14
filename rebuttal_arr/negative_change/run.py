#!/usr/bin/env python3
"""Standalone runner for the practical-negative WLLM rebuttal experiment."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping

from aggregate import aggregate_record_rows, existing_normal_rows
from obfuscators import (
    OBFUSCATOR_NAMES,
    available_obfuscators,
    obfuscate_all,
    obfuscate_python,
    obfuscator_versions,
)
from scorer import (
    DEFAULT_TOKENIZER,
    WllmConfigScorer,
    check_saved_score,
    load_tokenizer,
)
from source_data import (
    MODEL_NAME,
    REPO_ROOT,
    SUPPORTED_DATASETS,
    SourceDataError,
    WllmConfig,
    index_by_task,
    load_dataset_inputs,
    load_generate_records,
    load_obfuscation_records,
    read_jsonl,
    relative_to_repo,
    validate_inputs,
    write_jsonl,
)


HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE / "data"
NEGATIVE_CORPUS_ROOT = DATA_ROOT / "negative_corpus"
RECORD_ROOT = DATA_ROOT / "records"
MANIFEST_PATH = DATA_ROOT / "manifest.json"
METRICS_NEW_PATH = DATA_ROOT / "metrics_new.jsonl"
METRICS_NORMAL_PATH = DATA_ROOT / "metrics_normal_existing.jsonl"

POSITIVE_VARIANTS = {
    "clean_wllm": None,
    "pyminify_wllm": "pyminify",
    "pyminifier_wllm": "pyminifier",
}
NEGATIVE_VARIANTS = (
    "clean_no_wm_llm",
    "benchmark_reference",
    "pyminify_no_wm_llm",
    "pyminifier_no_wm_llm",
)

TASKS_PER_SCORE_SHARD = 30


@dataclass(frozen=True)
class ScoreShard:
    """One independently writable slice of a dataset/config scoring job."""

    global_index: int
    dataset: str
    config_id: str
    part_index: int
    task_start: int
    task_stop: int

    @property
    def task_count(self) -> int:
        return self.task_stop - self.task_start

    @property
    def output_path(self) -> Path:
        return (
            RECORD_ROOT
            / self.dataset
            / self.config_id
            / f"part-{self.part_index:03d}.jsonl"
        )


def _task_shard_ranges(task_count: int) -> tuple[tuple[int, int], ...]:
    if task_count <= 0:
        raise ValueError(f"task_count must be positive, got {task_count}")
    return tuple(
        (start, min(start + TASKS_PER_SCORE_SHARD, task_count))
        for start in range(0, task_count, TASKS_PER_SCORE_SHARD)
    )


def build_score_shards() -> list[ScoreShard]:
    """Build the stable global array order: dataset, config, then task slice."""

    shards: list[ScoreShard] = []
    for dataset in SUPPORTED_DATASETS:
        inputs = load_dataset_inputs(dataset)
        task_ranges = _task_shard_ranges(len(inputs.task_names))
        for config in inputs.wllm_configs:
            for part_index, (task_start, task_stop) in enumerate(task_ranges):
                shards.append(
                    ScoreShard(
                        global_index=len(shards),
                        dataset=dataset,
                        config_id=config.config_id,
                        part_index=part_index,
                        task_start=task_start,
                        task_stop=task_stop,
                    )
                )
    return shards


def _dataset_selection(value: str) -> tuple[str, ...]:
    return SUPPORTED_DATASETS if value == "all" else (value,)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_name = handle.name
            json.dump(dict(value), handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        if temp_name is not None:
            Path(temp_name).unlink(missing_ok=True)
        raise


def _package_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in ("torch", "transformers", "scipy", "scikit-learn", "nltk"):
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = None
    return result


def _git_commit() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _input_files(validation_report: Mapping[str, Any]) -> list[Path]:
    paths: set[Path] = set()
    for dataset_info in validation_report["datasets"].values():
        paths.add(REPO_ROOT / dataset_info["reference_path"])
        paths.add(REPO_ROOT / dataset_info["no_wm_run"]["generate_path"])
        for config in dataset_info["wllm_configs"]:
            paths.add(REPO_ROOT / config["generate_path"])
            paths.add(REPO_ROOT / config["obfuscate_path"])
            metrics_path = (REPO_ROOT / config["directory"]) / "metrics.jsonl"
            if metrics_path.is_file():
                paths.add(metrics_path)
    return sorted(paths)


def build_manifest(
    validation_report: Mapping[str, Any],
    tokenizer_name_or_path: str,
) -> dict[str, Any]:
    files = _input_files(validation_report)
    score_shards = build_score_shards()
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_NAME,
        "watermark": "wllm",
        "tokenizer": tokenizer_name_or_path,
        "git_commit": _git_commit(),
        "versions": {
            **_package_versions(),
            **obfuscator_versions(),
            "python": sys.version.split()[0],
        },
        "environment": {
            "python_executable": sys.executable,
            "obfuscators": available_obfuscators(),
        },
        "experiment": validation_report,
        "inputs": {
            relative_to_repo(path): {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in files
        },
        "negative_construction": {
            "clean_no_wm_llm": "saved no-WM g4d at temperature=1.0",
            "benchmark_reference": {
                "humaneval_py": "prompt + canonical_solution",
                "mbpp_py": "canonical_solution",
            },
            "pyminify_no_wm_llm": "pyminify(no-WM solution)",
            "pyminifier_no_wm_llm": "pyminifier(no-WM solution)",
        },
        "score_sharding": {
            "max_tasks_per_job": TASKS_PER_SCORE_SHARD,
            "job_count": len(score_shards),
            "dataset_job_counts": {
                dataset: sum(
                    shard.dataset == dataset for shard in score_shards
                )
                for dataset in SUPPORTED_DATASETS
            },
            "order": ["dataset", "config_id", "part_index"],
        },
        "metrics": ["auroc"],
    }


def validate_runtime(tokenizer_name_or_path: str) -> dict[str, Any]:
    validation_report = validate_inputs()
    availability = available_obfuscators()
    missing = [
        name for name, status in availability.items() if not status["executable"]
    ]
    if missing:
        raise RuntimeError(f"Missing obfuscator executables: {', '.join(missing)}")

    tokenizer = load_tokenizer(tokenizer_name_or_path, local_files_only=True)
    regressions: dict[str, Any] = {}
    for dataset in SUPPORTED_DATASETS:
        inputs = load_dataset_inputs(dataset)
        config = inputs.wllm_configs[0]
        generation = load_generate_records(config.generate_path)
        first_task = inputs.task_names[0]
        check = check_saved_score(generation[first_task], tokenizer=tokenizer)
        if not check["matches"]:
            raise RuntimeError(
                f"Saved-score regression failed for {dataset}/{config.config_id}/{first_task}: {check}"
            )
        regressions[dataset] = {"config": config.config_id, "task": first_task, **check}

    smoke_code = "def add_one(value):\n    return value + 1\n"
    obfuscator_smoke = {}
    for name in OBFUSCATOR_NAMES:
        transformed = obfuscate_python(smoke_code, name)
        if not transformed.ok:
            raise RuntimeError(f"{name} smoke test failed: {transformed.to_dict()}")
        obfuscator_smoke[name] = {
            "output_bytes": len((transformed.code or "").encode("utf-8")),
            "returncode": transformed.returncode,
        }

    return {
        "inputs": validation_report,
        "tokenizer": {
            "name_or_path": tokenizer.name_or_path,
            "vocab_size": len(tokenizer),
            "eos_token_id": tokenizer.eos_token_id,
            "local_files_only": True,
        },
        "obfuscators": availability,
        "obfuscator_smoke": obfuscator_smoke,
        "saved_score_regression": regressions,
    }


def command_validate(args: argparse.Namespace) -> None:
    report = validate_runtime(args.tokenizer_path)
    manifest = build_manifest(report["inputs"], args.tokenizer_path)
    if MANIFEST_PATH.exists() and not args.overwrite:
        print(f"validated; manifest already exists, left unchanged: {MANIFEST_PATH}")
    else:
        _atomic_json(MANIFEST_PATH, manifest, overwrite=args.overwrite)
        print(f"validated; wrote {MANIFEST_PATH}")
    print(json.dumps(report, indent=2))


def _corpus_path(dataset: str) -> Path:
    return NEGATIVE_CORPUS_ROOT / f"{dataset}.jsonl"


def prepare_dataset(dataset: str, *, overwrite: bool, timeout_seconds: float) -> None:
    output_path = _corpus_path(dataset)
    if output_path.exists() and not overwrite:
        print(f"skip existing corpus: {output_path}")
        return

    inputs = load_dataset_inputs(dataset)
    no_wm = load_generate_records(inputs.no_wm_run.generate_path)
    rows = []
    total = len(inputs.task_names)
    for index, task_name in enumerate(inputs.task_names, start=1):
        source = no_wm[task_name]
        reference = inputs.references[task_name]
        transformed = obfuscate_all(
            source["solution"], timeout_seconds=timeout_seconds
        )

        negative: dict[str, Any] = {
            "clean_no_wm_llm": {
                "g4d": source["g4d"],
                "solution": source["solution"],
            },
            "benchmark_reference": {
                "g4d": reference.g4d,
                "solution": reference.solution,
            },
        }
        errors: dict[str, Any] = {}
        audit: dict[str, Any] = {}
        for name, result in transformed.items():
            variant = f"{name}_no_wm_llm"
            audit[name] = {
                key: value
                for key, value in result.to_dict().items()
                if key != "code"
            }
            if result.ok:
                negative[variant] = {
                    "g4d": result.code,
                    "solution": result.code,
                }
            else:
                negative[variant] = {"g4d": None, "solution": None}
                errors[variant] = {
                    "code": result.error_code,
                    "message": result.error_message,
                }

        rows.append(
            {
                "dataset": dataset,
                "task": task_name,
                "negative": negative,
                "source": {
                    "no_wm_generate": relative_to_repo(inputs.no_wm_run.generate_path),
                    "no_wm_record_id": source["id"],
                    "no_wm_passed": source.get("passed"),
                    "reference_task_id": reference.source_task_id,
                },
                "obfuscation": audit,
                "errors": errors,
            }
        )
        if index % 25 == 0 or index == total:
            print(f"prepare {dataset}: {index}/{total}")

    write_jsonl(output_path, rows, overwrite=overwrite)
    print(f"wrote corpus: {output_path}")


def command_prepare(args: argparse.Namespace) -> None:
    for dataset in _dataset_selection(args.dataset):
        prepare_dataset(
            dataset,
            overwrite=args.overwrite,
            timeout_seconds=args.timeout,
        )


def _load_corpus(dataset: str) -> dict[str, dict[str, Any]]:
    path = _corpus_path(dataset)
    if not path.is_file():
        raise FileNotFoundError(f"Negative corpus is missing; run prepare first: {path}")
    return index_by_task(
        read_jsonl(path), task_field="task", context=str(path)
    )


def _positive_variant(
    generation: Mapping[str, Any],
    obfuscations: Mapping[str, Mapping[str, Any]],
    name: str,
) -> dict[str, Any]:
    obfuscator = POSITIVE_VARIANTS[name]
    if obfuscator is None:
        record = generation
    else:
        record = obfuscations.get(obfuscator)
        if record is None:
            return {"g4d": None, "solution": None, "z_score": None}
    return {
        "g4d": record.get("g4d"),
        "solution": record.get("solution"),
        "z_score": record.get("z_score"),
    }


def _config_by_id(dataset: str, config_id: str) -> WllmConfig:
    inputs = load_dataset_inputs(dataset)
    for config in inputs.wllm_configs:
        if config.config_id == config_id:
            return config
    raise SourceDataError(f"Unknown config {config_id!r} for {dataset}")


def _print_shard_result(
    shard: ScoreShard,
    *,
    status: str,
    row_count: int,
) -> None:
    print(
        "SHARD_RESULT "
        + json.dumps(
            {
                "status": status,
                "job_index": shard.global_index,
                "dataset": shard.dataset,
                "config": shard.config_id,
                "part": shard.part_index,
                "task_start": shard.task_start,
                "task_stop": shard.task_stop,
                "task_count": row_count,
                "output": str(shard.output_path),
            },
            sort_keys=True,
        )
    )


def score_dataset_shard(
    shard: ScoreShard,
    *,
    tokenizer: Any,
    tokenizer_name_or_path: str,
    overwrite: bool,
) -> None:
    dataset = shard.dataset
    config_id = shard.config_id
    output_path = shard.output_path
    if output_path.exists() and not overwrite:
        print(f"skip existing scores: {output_path}")
        _print_shard_result(
            shard,
            status="skipped_existing",
            row_count=len(read_jsonl(output_path)),
        )
        return

    inputs = load_dataset_inputs(dataset)
    config = _config_by_id(dataset, config_id)
    generation = load_generate_records(config.generate_path)
    positive_obfuscations = load_obfuscation_records(config)
    corpus = _load_corpus(dataset)
    no_wm = load_generate_records(inputs.no_wm_run.generate_path)
    config_scorer = WllmConfigScorer(
        gamma=config.gamma,
        ngram_len=config.ngram_len,
        tokenizer=tokenizer,
    )

    task_names = inputs.task_names[shard.task_start : shard.task_stop]
    if len(task_names) != shard.task_count:
        raise SourceDataError(
            f"Shard task range is invalid for {dataset}/{config_id}: "
            f"[{shard.task_start}, {shard.task_stop}) over {len(inputs.task_names)} tasks"
        )

    rows = []
    total = len(task_names)
    for index, task_name in enumerate(task_names, start=1):
        gen = generation[task_name]
        if gen["p4d"] != no_wm[task_name]["p4d"]:
            raise SourceDataError(f"p4d mismatch for {dataset}/{config_id}/{task_name}")
        corpus_row = corpus[task_name]
        positive = {
            name: _positive_variant(
                gen, positive_obfuscations.get(task_name, {}), name
            )
            for name in POSITIVE_VARIANTS
        }

        negative: dict[str, Any] = {}
        errors: dict[str, Any] = {}
        scoreable: dict[str, str] = {}
        for name in NEGATIVE_VARIANTS:
            code = corpus_row["negative"].get(name, {})
            g4d = code.get("g4d")
            if isinstance(g4d, str):
                scoreable[name] = g4d
            else:
                negative[name] = {
                    "corpus_task": task_name,
                    "z_score": None,
                    "invalid": None,
                    "num_tokens_scored": None,
                }
                errors[name] = corpus_row.get("errors", {}).get(
                    name, {"code": "negative_code_missing"}
                )

        scored = config_scorer.score_many(
            gen["p4d"], scoreable, custom_seed=gen["custom_seed"]
        )
        for name, score in scored.items():
            negative[name] = {
                "corpus_task": task_name,
                "z_score": score["z_score"],
                "invalid": score["invalid"],
                "num_tokens_scored": score["num_tokens_scored"],
            }

        rows.append(
            {
                "dataset": dataset,
                "config": config_id,
                "task": task_name,
                "score_shard": {
                    "job_index": shard.global_index,
                    "part": shard.part_index,
                    "task_start": shard.task_start,
                    "task_stop": shard.task_stop,
                },
                "detector": {
                    "tokenizer": tokenizer_name_or_path,
                    "custom_seed": gen["custom_seed"],
                    "gamma": config.gamma,
                    "delta": config.delta,
                    "temperature": config.temperature,
                    "ngram_len": config.ngram_len,
                    "p4d": gen["p4d"],
                },
                "positive": positive,
                "negative": negative,
                "errors": errors,
            }
        )
        if index % 20 == 0 or index == total:
            print(
                f"score job={shard.global_index} "
                f"{dataset}/{config_id}/part-{shard.part_index:03d}: "
                f"{index}/{total}"
            )

    write_jsonl(output_path, rows, overwrite=overwrite)
    print(f"wrote records: {output_path}")
    _print_shard_result(shard, status="completed", row_count=len(rows))


def _score_jobs(args: argparse.Namespace) -> list[ScoreShard]:
    jobs = build_score_shards()
    if args.all:
        if args.dataset is not None or args.config is not None or args.shard_index is not None:
            raise ValueError("--all cannot be combined with dataset/config shard selectors.")
        return jobs
    if args.job_index is not None:
        if args.dataset is not None or args.config is not None or args.shard_index is not None:
            raise ValueError("--job-index cannot be combined with dataset/config shard selectors.")
        if args.job_index < 0 or args.job_index >= len(jobs):
            raise ValueError(f"job-index must be in [0, {len(jobs) - 1}]")
        return [jobs[args.job_index]]
    if args.dataset is None or args.config is None:
        raise ValueError("Specify --all, --job-index, or both --dataset and --config.")
    selected = [
        shard
        for shard in jobs
        if shard.dataset == args.dataset and shard.config_id == args.config
    ]
    if not selected:
        raise SourceDataError(f"Unknown config {args.config!r} for {args.dataset}")
    if args.shard_index is not None:
        selected = [
            shard for shard in selected if shard.part_index == args.shard_index
        ]
        if not selected:
            part_count = len(
                {
                    shard.part_index
                    for shard in jobs
                    if shard.dataset == args.dataset
                    and shard.config_id == args.config
                }
            )
            raise ValueError(f"shard-index must be in [0, {part_count - 1}]")
    return selected


def command_score(args: argparse.Namespace) -> None:
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            f"Manifest is missing; run validate before scoring: {MANIFEST_PATH}"
        )
    with MANIFEST_PATH.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    validated_tokenizer = manifest.get("tokenizer")
    if validated_tokenizer != args.tokenizer_path:
        raise ValueError(
            "Tokenizer differs from the validated manifest: "
            f"validated={validated_tokenizer!r}, requested={args.tokenizer_path!r}. "
            "Run validate --overwrite with the intended tokenizer first."
        )
    tokenizer = load_tokenizer(args.tokenizer_path, local_files_only=True)
    for shard in _score_jobs(args):
        score_dataset_shard(
            shard,
            tokenizer=tokenizer,
            tokenizer_name_or_path=args.tokenizer_path,
            overwrite=args.overwrite,
        )


def load_config_score_rows(
    inputs: Any,
    config: WllmConfig,
    shards: list[ScoreShard],
) -> list[dict[str, Any]]:
    """Load one config's shards and require exact, duplicate-free task coverage."""

    config_id = config.config_id
    if not shards:
        raise ValueError(f"No score shards planned for {inputs.dataset}/{config_id}.")

    expected_paths = {shard.output_path for shard in shards}
    config_directory = RECORD_ROOT / inputs.dataset / config_id
    actual_paths = (
        set(config_directory.glob("part-*.jsonl"))
        if config_directory.is_dir()
        else set()
    )
    extra_paths = sorted(actual_paths - expected_paths)
    if extra_paths:
        raise ValueError(
            f"Unexpected score shard(s) for {inputs.dataset}/{config_id}: "
            f"{[str(path) for path in extra_paths]}"
        )

    combined: list[dict[str, Any]] = []
    task_owner: dict[str, Path] = {}
    for shard in sorted(shards, key=lambda item: item.part_index):
        record_path = shard.output_path
        if not record_path.is_file():
            raise FileNotFoundError(
                f"Missing score shard job={shard.global_index}: {record_path}"
            )
        rows = read_jsonl(record_path)
        actual_tasks = [row.get("task") for row in rows]
        duplicates = sorted(
            {task for task in actual_tasks if actual_tasks.count(task) > 1},
            key=str,
        )
        if duplicates:
            raise ValueError(f"Duplicate tasks in {record_path}: {duplicates}")

        expected_tasks = inputs.task_names[shard.task_start : shard.task_stop]
        actual_set = set(actual_tasks)
        expected_set = set(expected_tasks)
        missing = sorted(expected_set - actual_set)
        unexpected = sorted(actual_set - expected_set, key=str)
        if missing or unexpected or len(rows) != len(expected_tasks):
            raise ValueError(
                f"Task coverage mismatch in {record_path}: "
                f"expected={len(expected_tasks)}, actual={len(rows)}, "
                f"missing={missing}, unexpected={unexpected}"
            )

        for row in rows:
            if row.get("dataset") != inputs.dataset or row.get("config") != config_id:
                raise ValueError(
                    f"Wrong dataset/config row in {record_path}: "
                    f"{row.get('dataset')}/{row.get('config')}"
                )
            expected_detector = {
                "delta": config.delta,
                "gamma": config.gamma,
                "temperature": config.temperature,
                "ngram_len": config.ngram_len,
            }
            actual_detector = row.get("detector", {})
            mismatched_detector = {
                name: {
                    "expected": expected,
                    "actual": actual_detector.get(name),
                }
                for name, expected in expected_detector.items()
                if actual_detector.get(name) != expected
            }
            if mismatched_detector:
                raise ValueError(
                    f"Detector config mismatch for task {row.get('task')!r} "
                    f"in {record_path}: {mismatched_detector}"
                )
            expected_shard_metadata = {
                "job_index": shard.global_index,
                "part": shard.part_index,
                "task_start": shard.task_start,
                "task_stop": shard.task_stop,
            }
            if row.get("score_shard") != expected_shard_metadata:
                raise ValueError(
                    f"Score shard metadata mismatch for task {row.get('task')!r} "
                    f"in {record_path}: expected={expected_shard_metadata}, "
                    f"actual={row.get('score_shard')}"
                )
            task = row["task"]
            previous = task_owner.get(task)
            if previous is not None:
                raise ValueError(
                    f"Task {task!r} appears in both {previous} and {record_path}."
                )
            task_owner[task] = record_path
        combined.extend(rows)

    expected_all = set(inputs.task_names)
    actual_all = set(task_owner)
    if actual_all != expected_all or len(combined) != len(inputs.task_names):
        raise ValueError(
            f"Incomplete score coverage for {inputs.dataset}/{config_id}: "
            f"expected={len(inputs.task_names)}, actual={len(combined)}, "
            f"missing={sorted(expected_all - actual_all)}, "
            f"unexpected={sorted(actual_all - expected_all)}"
        )
    return combined


def command_aggregate(args: argparse.Namespace) -> None:
    new_rows = []
    normal_rows = []
    shards_by_config: dict[tuple[str, str], list[ScoreShard]] = {}
    for shard in build_score_shards():
        shards_by_config.setdefault((shard.dataset, shard.config_id), []).append(shard)
    for dataset in SUPPORTED_DATASETS:
        inputs = load_dataset_inputs(dataset)
        for config in inputs.wllm_configs:
            rows = load_config_score_rows(
                inputs,
                config,
                shards_by_config[(dataset, config.config_id)],
            )
            new_rows.extend(aggregate_record_rows(rows))
            if args.include_existing_normal:
                normal_rows.extend(
                    existing_normal_rows(
                        dataset,
                        config.config_id,
                        config.directory / "metrics.jsonl",
                    )
                )

    write_jsonl(METRICS_NEW_PATH, new_rows, overwrite=args.overwrite)
    print(f"wrote {len(new_rows)} new AUROC rows: {METRICS_NEW_PATH}")
    if args.include_existing_normal:
        write_jsonl(METRICS_NORMAL_PATH, normal_rows, overwrite=args.overwrite)
        print(f"wrote {len(normal_rows)} existing normal rows: {METRICS_NORMAL_PATH}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser(
        "validate", help="validate inputs, runtime, tools, tokenizer, and saved-score parity"
    )
    validate_parser.add_argument("--tokenizer-path", default=DEFAULT_TOKENIZER)
    validate_parser.add_argument("--overwrite", action="store_true")
    validate_parser.set_defaults(func=command_validate)

    prepare_parser = subparsers.add_parser(
        "prepare", help="construct and cache the four negative code distributions"
    )
    prepare_parser.add_argument(
        "--dataset", choices=("all", *SUPPORTED_DATASETS), default="all"
    )
    prepare_parser.add_argument("--timeout", type=float, default=10.0)
    prepare_parser.add_argument("--overwrite", action="store_true")
    prepare_parser.set_defaults(func=command_prepare)

    score_parser = subparsers.add_parser(
        "score", help="score all four negatives under saved WLLM task keys"
    )
    selector = score_parser.add_mutually_exclusive_group()
    selector.add_argument("--all", action="store_true")
    selector.add_argument("--job-index", type=int)
    score_parser.add_argument("--dataset", choices=SUPPORTED_DATASETS)
    score_parser.add_argument("--config")
    score_parser.add_argument(
        "--shard-index",
        type=int,
        help="0-based part within --dataset/--config; omitted means all its parts",
    )
    score_parser.add_argument("--tokenizer-path", default=DEFAULT_TOKENIZER)
    score_parser.add_argument("--overwrite", action="store_true")
    score_parser.set_defaults(func=command_score)

    aggregate_parser = subparsers.add_parser(
        "aggregate", help="build the complete 4-negative by 3-positive AUROC matrix"
    )
    aggregate_parser.add_argument("--include-existing-normal", action="store_true")
    aggregate_parser.add_argument("--overwrite", action="store_true")
    aggregate_parser.set_defaults(func=command_aggregate)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        args.func(args)
    except (SourceDataError, FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
