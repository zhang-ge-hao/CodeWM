#!/usr/bin/env python3
"""CLI for the filtered 100-step watermark-robustness experiment."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Mapping

import numpy as np
from scipy.stats import norm
from sklearn.metrics import auc

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from experiment.common import (  # type: ignore
        atomic_json,
        atomic_jsonl,
        build_manifest,
        index_by_task,
        iter_jsonl,
        load_manifest,
        load_transforms_for_config,
        resolve_repo_path,
        run_root,
        RULE_PROFILES,
        scheme_configs,
    )
else:
    from .common import (
        atomic_json,
        atomic_jsonl,
        build_manifest,
        index_by_task,
        iter_jsonl,
        load_manifest,
        load_transforms_for_config,
        resolve_repo_path,
        run_root,
        RULE_PROFILES,
        scheme_configs,
    )


DEFAULT_RUN_ID = "rw100-useful-v1"


def paper_auroc(samples: list[float]) -> float:
    """Reproduce ``src.metrics.cal_ideal_auroc`` exactly."""

    if not samples:
        raise ValueError("cannot calculate AUROC without positive scores")
    values = np.asarray(samples, dtype=float)
    thresholds = np.linspace(-15, 15, 3000)
    tpr = [np.mean(values > threshold) for threshold in thresholds]
    fpr = 1 - norm(0, 1).cdf(thresholds)
    return float(auc(fpr, tpr))


def _config_records(config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = list(iter_jsonl(resolve_repo_path(str(config["generate_path"]))))
    return rows, index_by_task(rows, str(config["generate_path"]))


def _check_saved_score(
    scorer: Any,
    config: Mapping[str, Any],
    record: Mapping[str, Any],
) -> dict[str, Any]:
    regression = scorer.score(record, str(record["g4d"]))
    saved_z = record.get("z_score")
    matches = saved_z is not None and math.isclose(
        float(saved_z),
        float(regression["z_score"]),
        rel_tol=0.0,
        abs_tol=1e-10,
    )
    if not matches:
        raise RuntimeError(
            f"saved-score regression failed for {config['key']}/{record['task_name']}: "
            f"saved={saved_z!r}, recomputed={regression['z_score']!r}"
        )
    label = "passed" if matches else "drift"
    print(
        f"saved-score regression {label}: {config['key']} task={record['task_name']} "
        f"saved={saved_z!r} recomputed={float(regression['z_score']):.12g}",
        flush=True,
    )
    return regression


def score_config(
    run_id: str,
    config: Mapping[str, Any],
    *,
    overwrite: bool,
    synthid_device: str = "cpu",
) -> Path:
    if __package__ in {None, ""}:
        from experiment.detectors import make_scorer  # type: ignore
    else:
        from .detectors import make_scorer

    output = run_root(run_id) / "scores" / f"{config['key']}.jsonl.gz"
    if output.exists() and not overwrite:
        print(f"skip existing score file: {output}")
        return output
    ordered, generations = _config_records(config)
    transforms = load_transforms_for_config(run_id, str(config["key"]))
    if set(generations) != set(transforms):
        raise ValueError(
            f"generation/transform task mismatch for {config['key']}: "
            f"{len(generations)} vs {len(transforms)}"
        )
    scorer = make_scorer(config, synthid_device=synthid_device)
    _check_saved_score(scorer, config, ordered[0])
    rows: list[dict[str, Any]] = []
    for position, generation in enumerate(ordered, start=1):
        task = str(generation["task_name"])
        transformed = transforms[task]
        common = {
            "schema_version": 1,
            "config_key": config["key"],
            "model_slug": config["model_slug"],
            "watermark": config["watermark"],
            "dataset": config["dataset"],
            "config_id": config["config_id"],
            "task_name": task,
            "record_id": generation["id"],
            "baseline_z_score": generation.get("z_score"),
            "saved_baseline_passed": generation.get("passed"),
            "baseline_passed": transformed.get("baseline", {})
            .get("execution", {})
            .get("passed"),
            "final_passed": transformed.get("final", {}).get("execution", {}).get("passed"),
            "transform_status": transformed.get("status"),
        }
        if transformed.get("status") != "ok":
            rows.append({**common, "status": "transform_error", "score": None})
            continue
        try:
            result = scorer.score(generation, str(transformed["detection_g4d"]))
            rows.append(
                {
                    **common,
                    "status": "ok",
                    "score": result,
                }
            )
        except Exception as error:
            rows.append(
                {
                    **common,
                    "status": "score_error",
                    "score": None,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
        if position % 25 == 0 or position == len(ordered):
            print(f"score {config['key']}: {position}/{len(ordered)}", flush=True)
    atomic_jsonl(output, rows, overwrite=overwrite)
    return output


def command_manifest(args: argparse.Namespace) -> None:
    manifest = build_manifest(
        run_id=args.run_id,
        steps=100,
        tasks_per_shard=args.tasks_per_shard,
        global_seed=args.global_seed,
        rule_profile=args.rule_profile,
    )
    output = run_root(args.run_id) / "manifest.json"
    atomic_json(output, manifest, overwrite=args.overwrite)
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))
    print(output)


def command_transform(args: argparse.Namespace) -> None:
    if __package__ in {None, ""}:
        from experiment.transform import transform_shard  # type: ignore
    else:
        from .transform import transform_shard

    index = args.index
    if index is None:
        value = os.environ.get("SLURM_ARRAY_TASK_ID")
        if value is None:
            raise ValueError("--index or SLURM_ARRAY_TASK_ID is required")
        index = int(value)
    output = transform_shard(
        args.run_id,
        index,
        timeout=args.timeout,
        memory_mb=args.memory_mb,
        overwrite=args.overwrite,
    )
    print(output)


def command_score_index(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.run_id)
    configs = scheme_configs(manifest, args.scheme)
    index = args.index
    if index is None:
        value = os.environ.get("SLURM_ARRAY_TASK_ID")
        if value is None:
            raise ValueError("--index or SLURM_ARRAY_TASK_ID is required")
        index = int(value)
    if index < 0 or index >= len(configs):
        raise IndexError(f"{args.scheme} config index {index} outside [0, {len(configs)})")
    print(
        score_config(
            args.run_id,
            configs[index],
            overwrite=args.overwrite,
            synthid_device=args.synthid_device,
        )
    )


def command_score_sweet_model(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.run_id)
    configs = [
        config
        for config in scheme_configs(manifest, "sweet")
        if config["model_slug"] == args.model_slug
    ]
    if not configs:
        raise ValueError(f"no selected SWEET configs for {args.model_slug}")
    for index, config in enumerate(configs, start=1):
        print(f"SWEET config {index}/{len(configs)}: {config['key']}", flush=True)
        score_config(args.run_id, config, overwrite=args.overwrite)


def _balanced_config_shards(
    configs: list[dict[str, Any]], shard_count: int
) -> list[list[dict[str, Any]]]:
    if shard_count < 1:
        raise ValueError("shard count must be positive")
    buckets: list[list[dict[str, Any]]] = [[] for _ in range(shard_count)]
    loads = [0] * shard_count
    for config in sorted(
        configs,
        key=lambda item: (-int(item["task_count"]), str(item["key"])),
    ):
        index = min(range(shard_count), key=lambda value: (loads[value], value))
        buckets[index].append(config)
        loads[index] += int(config["task_count"])
    return buckets


def command_score_sweet_shard(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.run_id)
    configs = [
        config
        for config in scheme_configs(manifest, "sweet")
        if config["model_slug"] == args.model_slug
    ]
    buckets = _balanced_config_shards(configs, args.shards)
    index = args.index
    if index is None:
        value = os.environ.get("SLURM_ARRAY_TASK_ID")
        if value is None:
            raise ValueError("--index or SLURM_ARRAY_TASK_ID is required")
        index = int(value)
    if index < 0 or index >= len(buckets):
        raise IndexError(f"SWEET shard {index} outside [0, {len(buckets)})")
    selected = buckets[index]
    if not selected:
        raise ValueError(f"SWEET shard {index} is empty")
    print(
        f"SWEET shard {index}/{len(buckets)}: "
        f"configs={len(selected)} tasks={sum(int(c['task_count']) for c in selected)}",
        flush=True,
    )
    for config in selected:
        score_config(args.run_id, config, overwrite=args.overwrite)


def _score_config_parallel_worker(
    run_id: str, config: dict[str, Any], overwrite: bool, synthid_device: str
) -> str:
    return str(
        score_config(
            run_id,
            config,
            overwrite=overwrite,
            synthid_device=synthid_device,
        )
    )


def command_score_scheme_parallel(args: argparse.Namespace) -> None:
    configs = scheme_configs(load_manifest(args.run_id), args.scheme)
    workers = min(args.workers, len(configs))
    failures: list[tuple[str, BaseException]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _score_config_parallel_worker,
                args.run_id,
                config,
                args.overwrite,
                args.synthid_device,
            ): str(config["key"])
            for config in configs
        }
        for future in as_completed(futures):
            key = futures[future]
            try:
                print(f"config={key} output={future.result()}", flush=True)
            except BaseException as error:
                failures.append((key, error))
                print(
                    f"config={key} failed={type(error).__name__}: {error}",
                    flush=True,
                )
    if failures:
        detail = "; ".join(
            f"{key}:{type(error).__name__}:{error}"
            for key, error in sorted(failures)
        )
        raise RuntimeError(f"{len(failures)} score configs failed: {detail}")


def command_validate_transforms(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.run_id)
    root = run_root(args.run_id)
    expected_shards = len(manifest["shards"])
    expected_rows = int(manifest["counts"]["walks"])
    paths = sorted((root / "transforms").glob("part-*.jsonl.gz"))
    if len(paths) != expected_shards:
        raise RuntimeError(
            f"transform shard count mismatch: {len(paths)} != {expected_shards}"
        )
    record_ids: set[str] = set()
    errors: list[str] = []
    violation_count = 0
    status_counts: Counter[str] = Counter()
    row_count = 0

    def violation(message: str) -> None:
        nonlocal violation_count
        violation_count += 1
        if len(errors) < args.max_errors:
            errors.append(message)

    def original_lacks_projected_entry(row: Mapping[str, Any]) -> bool:
        if row.get("error_type") != "ValueError":
            return False
        match = re.fullmatch(
            r"expected one top-level function '([^']+)', found 0",
            str(row.get("error")),
        )
        programs = row.get("programs", ())
        if match is None or not programs or not isinstance(programs[0], str):
            return False
        try:
            tree = ast.parse(programs[0])
        except SyntaxError:
            return False
        entry = match.group(1)
        return not any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == entry
            for node in tree.body
        )

    for shard_index, path in enumerate(paths):
        expected_name = f"part-{shard_index:04d}.jsonl.gz"
        if path.name != expected_name:
            violation(f"unexpected shard order: {path.name} != {expected_name}")
        for row in iter_jsonl(path):
            row_count += 1
            record_id = str(row.get("record_id"))
            if record_id in record_ids:
                violation(f"duplicate record_id: {record_id}")
            record_ids.add(record_id)
            status = str(row.get("status"))
            status_counts[status] += 1
            if status != "ok":
                # Saved model generations can themselves be syntactically
                # invalid.  They are a legitimate transform-error cohort only
                # when the engine failed before taking its first transition.
                initial_source_error = (
                    status == "error"
                    and len(row.get("programs", ())) == 1
                    and len(row.get("trace", ())) == 0
                    and bool(row.get("error_type"))
                )
                input_projection_error = (
                    status == "error"
                    and len(row.get("programs", ())) == 101
                    and len(row.get("trace", ())) == 100
                    and original_lacks_projected_entry(row)
                )
                if not (initial_source_error or input_projection_error):
                    violation(
                        f"{record_id}: non-initial transform failure "
                        f"status={status} programs={len(row.get('programs', ()))} "
                        f"trace={len(row.get('trace', ()))} "
                        f"error={row.get('error_type')}:{row.get('error')}"
                    )
                continue
            if len(row.get("programs", ())) != 101:
                violation(f"{record_id}: programs != 101")
            if len(row.get("trace", ())) != 100:
                violation(f"{record_id}: trace != 100")
            if not isinstance(row.get("detection_g4d"), str):
                violation(f"{record_id}: detection_g4d is missing")
            if not isinstance(row.get("final", {}).get("execution"), dict):
                violation(f"{record_id}: final execution is missing")
    if row_count != expected_rows:
        violation(f"transform row count mismatch: {row_count} != {expected_rows}")
    if violation_count:
        raise RuntimeError(
            f"transform validation failed with {violation_count} violations:\n"
            + "\n".join(errors)
        )
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "transform_shards": len(paths),
                "rows": row_count,
                "statuses": dict(sorted(status_counts.items())),
                "programs_per_row": 101,
                "trace_per_row": 100,
            },
            indent=2,
            sort_keys=True,
        )
    )


def command_validate_scorer(args: argparse.Namespace) -> None:
    if __package__ in {None, ""}:
        from experiment.detectors import make_scorer  # type: ignore
    else:
        from .detectors import make_scorer

    manifest = load_manifest(args.run_id)
    configs = scheme_configs(manifest, args.scheme)
    if args.index < 0 or args.index >= len(configs):
        raise IndexError(f"{args.scheme} config index {args.index} outside [0, {len(configs)})")
    config = configs[args.index]
    ordered, _ = _config_records(config)
    scorer = make_scorer(config, synthid_device=args.synthid_device)
    _check_saved_score(scorer, config, ordered[0])


def _metric_row(
    config: Mapping[str, Any],
    generations: Mapping[str, Mapping[str, Any]],
    transforms: Mapping[str, Mapping[str, Any]],
    scores: list[dict[str, Any]],
) -> dict[str, Any]:
    scored = [row for row in scores if row.get("status") == "ok" and row.get("score")]
    score_tasks = {str(row["task_name"]) for row in scored}
    final_z = [float(row["score"]["z_score"]) for row in scored]
    original_z_same_cohort = []
    for task in score_tasks:
        saved = generations[task].get("z_score")
        if saved is not None:
            original_z_same_cohort.append(float(saved))
    attempted = len(generations)
    transformed_ok = sum(row.get("status") == "ok" for row in transforms.values())
    final_passed = sum(
        row.get("final", {}).get("execution", {}).get("passed") is True
        for row in transforms.values()
    )
    baseline_pass_tasks = {
        task
        for task, row in transforms.items()
        if row.get("baseline", {}).get("execution", {}).get("passed") is True
    }
    preserved = sum(
        transforms[task].get("final", {}).get("execution", {}).get("passed") is True
        for task in baseline_pass_tasks
    )
    regressions = sum(
        transforms[task].get("final", {}).get("execution", {}).get("passed") is False
        for task in baseline_pass_tasks
    )
    rule_counts: Counter[str] = Counter()
    identity_steps = 0
    changed = 0
    for row in transforms.values():
        rule_counts.update({str(k): int(v) for k, v in row.get("rule_counts", {}).items()})
        identity_steps += int(row.get("rule_counts", {}).get("identity", 0))
        changed += int(row.get("final", {}).get("changed") is True)
    status_counts = Counter(str(row.get("status")) for row in transforms.values())
    score_status_counts = Counter(str(row.get("status")) for row in scores)
    return {
        "config_key": config["key"],
        "model_slug": config["model_slug"],
        "model_name": config["model_name"],
        "watermark": config["watermark"],
        "dataset": config["dataset"],
        "config_id": config["config_id"],
        "temperature": config["temperature"],
        "delta": config["delta"],
        "gamma": config["gamma"],
        "entropy_threshold": config["entropy_threshold"],
        "ngram_len": config["ngram_len"],
        "selection_original_auroc": config["original_auroc"],
        "selection_original_pass1": config["original_pass1"],
        "selection_no_wm_pass1": config["no_wm_pass1"],
        "selection_pass1_retention": config["pass1_retention"],
        "same_cohort_original_auroc": paper_auroc(original_z_same_cohort),
        "same_cohort_saved_original_auroc": paper_auroc(original_z_same_cohort),
        "baseline_recomputed": False,
        "baseline_z_abs_drift_mean": 0.0,
        "baseline_z_abs_drift_max": 0.0,
        "step100_auroc": paper_auroc(final_z),
        "step100_mean_z": float(np.mean(final_z)),
        "step100_detection_rate_z_gt_4": float(np.mean(np.asarray(final_z) > 4.0)),
        "attempted": attempted,
        "transform_rows": len(transforms),
        "score_rows": len(scores),
        "transformed_ok": transformed_ok,
        "scored": len(scored),
        "baseline_passed": len(baseline_pass_tasks),
        "final_passed": final_passed,
        "final_pass1": final_passed / attempted,
        "preserved_baseline_passed": preserved,
        "semantic_regressions": regressions,
        "conditional_pass_preservation": (
            preserved / len(baseline_pass_tasks) if baseline_pass_tasks else None
        ),
        "changed_outputs": changed,
        "identity_steps": identity_steps,
        "transform_status_counts": dict(sorted(status_counts.items())),
        "score_status_counts": dict(sorted(score_status_counts.items())),
        "rule_counts": dict(sorted(rule_counts.items())),
    }


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scalar_fields = [
        key
        for key, value in rows[0].items()
        if not isinstance(value, (dict, list, tuple))
    ]
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=scalar_fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key) for key in scalar_fields})
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def command_aggregate(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.run_id)
    metrics: list[dict[str, Any]] = []
    for config in manifest["configs"]:
        score_path = run_root(args.run_id) / "scores" / f"{config['key']}.jsonl.gz"
        if not score_path.is_file():
            raise FileNotFoundError(f"score file is missing: {score_path}")
        _, generations = _config_records(config)
        scores = list(iter_jsonl(score_path))
        metrics.append(
            _metric_row(
                config,
                generations,
                load_transforms_for_config(args.run_id, str(config["key"])),
                scores,
            )
        )
    if args.require_complete:
        incomplete = [
            row
            for row in metrics
            if int(row["transform_rows"]) != int(row["attempted"])
            or int(row["score_rows"]) != int(row["attempted"])
            or int(row["scored"]) != int(row["transformed_ok"])
            or set(row["score_status_counts"]) - {"ok", "transform_error"}
        ]
        if incomplete:
            detail = ", ".join(
                f"{row['config_key']}: attempted={row['attempted']} "
                f"transform_rows={row['transform_rows']} transformed_ok={row['transformed_ok']} "
                f"score_rows={row['score_rows']} scored={row['scored']} "
                f"score_statuses={row['score_status_counts']}"
                for row in incomplete
            )
            raise RuntimeError(f"incomplete formal experiment: {detail}")
    root = run_root(args.run_id)
    atomic_jsonl(root / "metrics.jsonl", metrics, overwrite=args.overwrite)
    _atomic_csv(root / "metrics.csv", metrics)
    scheme_summary: dict[str, Any] = {}
    for scheme in ("wllm", "sweet", "synthid"):
        rows = [row for row in metrics if row["watermark"] == scheme]
        aurocs = [float(row["step100_auroc"]) for row in rows]
        scheme_summary[scheme] = {
            "configs": len(rows),
            "walks": sum(int(row["attempted"]) for row in rows),
            "scored": sum(int(row["scored"]) for row in rows),
            "auroc_min": min(aurocs),
            "auroc_mean": float(np.mean(aurocs)),
            "auroc_max": max(aurocs),
            "configs_below_0_6": sum(value < 0.6 for value in aurocs),
            "semantic_regressions": sum(int(row["semantic_regressions"]) for row in rows),
            "transform_errors": sum(
                int(row["attempted"]) - int(row["transformed_ok"]) for row in rows
            ),
        }
    summary = {
        "run_id": args.run_id,
        "negative_distribution": "standard_normal",
        "configs": len(metrics),
        "walks": sum(int(row["attempted"]) for row in metrics),
        "scored": sum(int(row["scored"]) for row in metrics),
        "semantic_regressions": sum(int(row["semantic_regressions"]) for row in metrics),
        "transform_errors": sum(
            int(row["attempted"]) - int(row["transformed_ok"]) for row in metrics
        ),
        "by_scheme": scheme_summary,
    }
    atomic_json(root / "summary.json", summary, overwrite=args.overwrite)
    print(json.dumps(summary, indent=2, sort_keys=True))


def command_status(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.run_id)
    root = run_root(args.run_id)
    transform_expected = len(manifest["shards"])
    transform_present = len(list((root / "transforms").glob("part-*.jsonl.gz")))
    score_expected = len(manifest["configs"])
    score_present = len(list((root / "scores").glob("*.jsonl.gz")))
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "transform_shards": {"present": transform_present, "expected": transform_expected},
                "score_files": {"present": score_present, "expected": score_expected},
                "summary_exists": (root / "summary.json").is_file(),
            },
            indent=2,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser("manifest")
    manifest.add_argument("--tasks-per-shard", type=int, default=100)
    manifest.add_argument("--global-seed", type=int, default=10771)
    manifest.add_argument("--rule-profile", choices=RULE_PROFILES, default="full")
    manifest.add_argument("--overwrite", action="store_true")
    manifest.set_defaults(function=command_manifest)

    transform = subparsers.add_parser("transform")
    transform.add_argument("--index", type=int)
    transform.add_argument("--timeout", type=float, default=10.0)
    transform.add_argument("--memory-mb", type=int, default=1024)
    transform.add_argument("--overwrite", action="store_true")
    transform.set_defaults(function=command_transform)

    score = subparsers.add_parser("score-index")
    score.add_argument("--scheme", choices=("wllm", "synthid"), required=True)
    score.add_argument("--index", type=int)
    score.add_argument("--synthid-device", default="cpu")
    score.add_argument("--overwrite", action="store_true")
    score.set_defaults(function=command_score_index)

    sweet = subparsers.add_parser("score-sweet-model")
    sweet.add_argument("--model-slug", choices=("Llama31Instruct8B", "DSCoderBase33B"), required=True)
    sweet.add_argument("--overwrite", action="store_true")
    sweet.set_defaults(function=command_score_sweet_model)

    sweet_shard = subparsers.add_parser("score-sweet-shard")
    sweet_shard.add_argument("--model-slug", choices=("Llama31Instruct8B", "DSCoderBase33B"), required=True)
    sweet_shard.add_argument("--shards", type=int, required=True)
    sweet_shard.add_argument("--index", type=int)
    sweet_shard.add_argument("--overwrite", action="store_true")
    sweet_shard.set_defaults(function=command_score_sweet_shard)

    parallel_score = subparsers.add_parser("score-scheme-parallel")
    parallel_score.add_argument("--scheme", choices=("wllm",), required=True)
    parallel_score.add_argument("--workers", type=int, default=13)
    parallel_score.add_argument("--synthid-device", default="cpu")
    parallel_score.add_argument("--overwrite", action="store_true")
    parallel_score.set_defaults(function=command_score_scheme_parallel)

    validate_transforms = subparsers.add_parser("validate-transforms")
    validate_transforms.add_argument("--max-errors", type=int, default=20)
    validate_transforms.set_defaults(function=command_validate_transforms)

    validate_scorer = subparsers.add_parser("validate-scorer")
    validate_scorer.add_argument("--scheme", choices=("wllm", "synthid"), required=True)
    validate_scorer.add_argument("--index", type=int, default=0)
    validate_scorer.add_argument("--synthid-device", default="cpu")
    validate_scorer.set_defaults(function=command_validate_scorer)

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--overwrite", action="store_true")
    aggregate.add_argument("--require-complete", action="store_true")
    aggregate.set_defaults(function=command_aggregate)

    status = subparsers.add_parser("status")
    status.set_defaults(function=command_status)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.function(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
