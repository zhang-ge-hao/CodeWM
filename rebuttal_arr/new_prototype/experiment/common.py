"""Read-only input discovery and durable experiment I/O.

All paper-result inputs are treated as immutable.  New artifacts are confined
to ``new_prototype/data/watermark_attack/<run-id>``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile
import sys
from typing import Any, Iterable, Iterator, Mapping


NEW_PROTOTYPE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = NEW_PROTOTYPE_ROOT.parents[1]
RESULT_ROOT = REPO_ROOT / "data" / "result"
DATA_ROOT = NEW_PROTOTYPE_ROOT / "data" / "watermark_attack"

SUPPORTED_SCHEMES = ("wllm", "sweet", "synthid")
RULE_PROFILES = ("full", "no_advanced")
EXPECTED_TASKS = {"humaneval_py": 164, "mbpp_py": 378}
MODEL_SLUG_TO_NAME = {
    "Llama31Instruct8B": "meta-llama/Llama-3.1-8B-Instruct",
    "DSCoderBase33B": "deepseek-ai/deepseek-coder-33b-base",
}


class ExperimentInputError(ValueError):
    """Raised when saved paper results do not satisfy the frozen contract."""


@dataclass(frozen=True)
class SelectedConfig:
    key: str
    model_slug: str
    model_name: str
    watermark: str
    dataset: str
    config_id: str
    directory: str
    generate_path: str
    metrics_path: str
    task_count: int
    temperature: float
    delta: float | None
    gamma: float | None
    entropy_threshold: float | None
    ngram_len: int
    original_auroc: float
    original_pass1: float
    no_wm_pass1: float
    pass1_retention: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def relative_to_repo(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def resolve_repo_path(value: str) -> Path:
    path = (REPO_ROOT / value).resolve()
    path.relative_to(REPO_ROOT.resolve())
    return path


def run_root(run_id: str) -> Path:
    if not run_id or run_id in {".", ".."} or "/" in run_id or "\\" in run_id:
        raise ValueError(f"invalid run id: {run_id!r}")
    path = (DATA_ROOT / run_id).resolve()
    path.relative_to(DATA_ROOT.resolve())
    return path


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ExperimentInputError(
                    f"expected object at {path}:{line_number}, got {type(value).__name__}"
                )
            yield value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def index_by_task(rows: Iterable[Mapping[str, Any]], context: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        task = row.get("task_name")
        if not isinstance(task, str) or not task:
            raise ExperimentInputError(f"missing task_name in {context}")
        if task in result:
            raise ExperimentInputError(f"duplicate task {task!r} in {context}")
        result[task] = dict(row)
    return result


def atomic_json(path: Path, value: Mapping[str, Any], *, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(dict(value), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def atomic_jsonl(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    overwrite: bool = False,
) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        suffix = ".tmp.gz" if path.suffix == ".gz" else ".tmp"
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=suffix,
            delete=False,
        ) as raw:
            temporary = Path(raw.name)
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(temporary, "wt", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")))
                handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(global_seed: int, record_id: str) -> int:
    payload = f"{int(global_seed)}\0{record_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _original_metric(path: Path) -> dict[str, Any]:
    originals = [row for row in iter_jsonl(path) if row.get("obf_name") == "Original"]
    if len(originals) != 1:
        raise ExperimentInputError(f"expected one Original row in {path}, found {len(originals)}")
    return originals[0]


def _parse_result_directory(directory: Path) -> tuple[str, str, str]:
    pieces = directory.parent.name.split("--")
    if len(pieces) != 3:
        raise ExperimentInputError(f"unexpected result directory: {directory}")
    return pieces[0], pieces[1], pieces[2]


def _no_wm_pass1() -> dict[tuple[str, str, float], float]:
    result: dict[tuple[str, str, float], float] = {}
    for metrics_path in sorted(RESULT_ROOT.glob("*--no_wm--*_py/*/metrics.jsonl")):
        row = _original_metric(metrics_path)
        key = (str(row["model_name"]), str(row["dataset_name"]), float(row["temperature"]))
        value = float(row["pass1"])
        previous = result.setdefault(key, value)
        if previous != value:
            raise ExperimentInputError(f"conflicting no-WM Pass@1 for {key}: {previous} vs {value}")
    return result


def discover_selected_configs(
    *,
    auroc_threshold: float = 0.8,
    pass1_retention_threshold: float = 0.8,
) -> list[SelectedConfig]:
    """Select useful original-watermark configurations before observing attack results."""

    baselines = _no_wm_pass1()
    selected: list[SelectedConfig] = []
    for metrics_path in sorted(RESULT_ROOT.glob("*--*--*_py/*/metrics.jsonl")):
        directory = metrics_path.parent
        model_slug, watermark, dataset = _parse_result_directory(directory)
        if watermark not in SUPPORTED_SCHEMES or dataset not in EXPECTED_TASKS:
            continue
        row = _original_metric(metrics_path)
        model_name = str(row["model_name"])
        if MODEL_SLUG_TO_NAME.get(model_slug) != model_name:
            raise ExperimentInputError(f"model slug/name mismatch in {metrics_path}")
        temperature = float(row["temperature"])
        baseline_key = (model_name, dataset, temperature)
        if baseline_key not in baselines:
            raise ExperimentInputError(f"missing no-WM baseline for {baseline_key}")
        no_wm_pass1 = baselines[baseline_key]
        original_auroc = float(row["auroc"])
        original_pass1 = float(row["pass1"])
        if not (
            original_auroc > float(auroc_threshold)
            and original_pass1 > float(pass1_retention_threshold) * no_wm_pass1
        ):
            continue
        generate_path = directory / "generate.jsonl"
        task_count = sum(1 for _ in iter_jsonl(generate_path))
        expected = EXPECTED_TASKS[dataset]
        if task_count != expected:
            raise ExperimentInputError(
                f"expected {expected} generation records in {generate_path}, found {task_count}"
            )
        config_id = directory.name
        selected.append(
            SelectedConfig(
                key=f"{model_slug}--{watermark}--{dataset}--{config_id}",
                model_slug=model_slug,
                model_name=model_name,
                watermark=watermark,
                dataset=dataset,
                config_id=config_id,
                directory=relative_to_repo(directory),
                generate_path=relative_to_repo(generate_path),
                metrics_path=relative_to_repo(metrics_path),
                task_count=task_count,
                temperature=temperature,
                delta=None if row.get("delta") is None else float(row["delta"]),
                gamma=None if row.get("gamma") is None else float(row["gamma"]),
                entropy_threshold=(
                    None
                    if row.get("entropy_threshold") is None
                    else float(row["entropy_threshold"])
                ),
                ngram_len=int(row["ngram_len"]),
                original_auroc=original_auroc,
                original_pass1=original_pass1,
                no_wm_pass1=no_wm_pass1,
                pass1_retention=original_pass1 / no_wm_pass1,
            )
        )
    return selected


def build_manifest(
    *,
    run_id: str,
    steps: int = 100,
    tasks_per_shard: int = 100,
    global_seed: int = 10771,
    rule_profile: str = "full",
) -> dict[str, Any]:
    if steps != 100:
        raise ValueError("the frozen experiment endpoint is exactly 100 steps")
    if tasks_per_shard <= 0:
        raise ValueError("tasks_per_shard must be positive")
    if rule_profile not in RULE_PROFILES:
        raise ValueError(
            f"unknown rule profile {rule_profile!r}; expected one of {RULE_PROFILES}"
        )
    configs = discover_selected_configs()
    shards: list[dict[str, Any]] = []
    for config_index, config in enumerate(configs):
        for start in range(0, config.task_count, tasks_per_shard):
            shards.append(
                {
                    "index": len(shards),
                    "config_index": config_index,
                    "config_key": config.key,
                    "task_start": start,
                    "task_stop": min(start + tasks_per_shard, config.task_count),
                }
            )
    scheme_counts = {
        scheme: sum(config.watermark == scheme for config in configs)
        for scheme in SUPPORTED_SCHEMES
    }
    walk_counts = {
        scheme: sum(config.task_count for config in configs if config.watermark == scheme)
        for scheme in SUPPORTED_SCHEMES
    }
    expected_scheme_counts = {"wllm": 13, "sweet": 24, "synthid": 6}
    expected_walk_counts = {"wllm": 3416, "sweet": 6076, "synthid": 1626}
    if scheme_counts != expected_scheme_counts or walk_counts != expected_walk_counts:
        raise ExperimentInputError(
            "selected input universe changed: "
            f"configs={scheme_counts}, walks={walk_counts}"
        )
    source_files = sorted(
        (NEW_PROTOTYPE_ROOT / "rw_obfuscator").rglob("*.py")
    )
    input_files = sorted(
        {
            resolve_repo_path(config.generate_path)
            for config in configs
        }
        | {
            resolve_repo_path(config.metrics_path)
            for config in configs
        }
        | set(RESULT_ROOT.glob("*--no_wm--*_py/*/metrics.jsonl"))
    )
    return {
        "schema_version": 1,
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection": {
            "original_auroc_strictly_above": 0.8,
            "original_pass1_retention_strictly_above": 0.8,
            "negative_distribution": "standard_normal",
        },
        "walk": {
            "steps": steps,
            "trajectories_per_input": 1,
            "global_seed": global_seed,
            "rule_profile": rule_profile,
            "save_every_program": True,
            "evaluate_steps": [100],
            "identity_is_uniform_concrete_action": True,
            "retry_or_resample": False,
        },
        "configs": [config.to_dict() for config in configs],
        "shards": shards,
        "counts": {
            "configs": len(configs),
            "configs_by_scheme": scheme_counts,
            "walks": sum(walk_counts.values()),
            "walks_by_scheme": walk_counts,
            "transitions": steps * sum(walk_counts.values()),
            "saved_programs": (steps + 1) * sum(walk_counts.values()),
            "transform_shards": len(shards),
        },
        "environment": {"python": sys.version.split()[0]},
        "input_files": {
            relative_to_repo(path): {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in input_files
        },
        "obfuscator_sources": {
            relative_to_repo(path): sha256_file(path) for path in source_files
        },
    }


def load_manifest(run_id: str) -> dict[str, Any]:
    path = run_root(run_id) / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"manifest is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("run_id") != run_id:
        raise ExperimentInputError(f"manifest run id mismatch in {path}")
    return value


def config_map(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(config["key"]): dict(config) for config in manifest["configs"]}


def scheme_configs(manifest: Mapping[str, Any], scheme: str) -> list[dict[str, Any]]:
    if scheme not in SUPPORTED_SCHEMES:
        raise ValueError(f"unsupported watermark: {scheme}")
    return [dict(config) for config in manifest["configs"] if config["watermark"] == scheme]


def load_transforms_for_config(run_id: str, config_key: str) -> dict[str, dict[str, Any]]:
    manifest = load_manifest(run_id)
    rows: list[dict[str, Any]] = []
    for shard in manifest["shards"]:
        if shard["config_key"] != config_key:
            continue
        path = run_root(run_id) / "transforms" / f"part-{int(shard['index']):04d}.jsonl.gz"
        if not path.is_file():
            raise FileNotFoundError(f"transform shard is missing: {path}")
        rows.extend(iter_jsonl(path))
    return index_by_task(rows, f"transforms for {config_key}")


__all__ = [
    "DATA_ROOT",
    "EXPECTED_TASKS",
    "ExperimentInputError",
    "MODEL_SLUG_TO_NAME",
    "NEW_PROTOTYPE_ROOT",
    "RULE_PROFILES",
    "REPO_ROOT",
    "RESULT_ROOT",
    "SUPPORTED_SCHEMES",
    "SelectedConfig",
    "atomic_json",
    "atomic_jsonl",
    "build_manifest",
    "config_map",
    "discover_selected_configs",
    "index_by_task",
    "iter_jsonl",
    "load_manifest",
    "load_transforms_for_config",
    "read_jsonl",
    "relative_to_repo",
    "resolve_repo_path",
    "run_root",
    "scheme_configs",
    "sha256_file",
    "stable_seed",
]
