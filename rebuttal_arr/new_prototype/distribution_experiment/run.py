#!/usr/bin/env python3
"""Sample rule-based equivalent spaces and test their z-score distributions."""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import csv
from datetime import datetime, timezone
from functools import lru_cache
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
from typing import Any, Iterable, Iterator, Mapping

from experiment.common import (
    EXPECTED_TASKS,
    MODEL_SLUG_TO_NAME,
    REPO_ROOT,
    RESULT_ROOT,
    atomic_json,
    atomic_jsonl,
    iter_jsonl,
    relative_to_repo,
    resolve_repo_path,
    sha256_file,
)


NEW_PROTOTYPE_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = NEW_PROTOTYPE_ROOT / "data" / "distribution_consistency"
SCHEMES = ("wllm", "sweet", "synthid")
RUN_ID = "rw100-z4-sample100-v2"
SAMPLE_SIZE = 100
TRAJECTORIES_PER_SEED = 30
STEPS = 100
GLOBAL_SEED = 10771
TASKS_PER_SHARD = 8
FOLLOWUP_STEPS = 500
USEFUL_SOURCE_RUN_ID = "rw100-useful-v3-full"
TRANSFORM_TIMEOUT_SECONDS = 5.0
TRANSFORM_MEMORY_MB = 768


def run_root(run_id: str) -> Path:
    if not run_id or run_id in {".", ".."} or "/" in run_id or "\\" in run_id:
        raise ValueError(f"invalid run id: {run_id!r}")
    result = (DATA_ROOT / run_id).resolve()
    result.relative_to(DATA_ROOT.resolve())
    return result


def _original_metric(path: Path) -> dict[str, Any]:
    rows = [row for row in iter_jsonl(path) if row.get("obf_name") == "Original"]
    if len(rows) != 1:
        raise ValueError(f"expected one Original row in {path}, found {len(rows)}")
    return rows[0]


def _config_from_metrics(metrics_path: Path) -> dict[str, Any]:
    directory = metrics_path.parent
    pieces = directory.parent.name.split("--")
    if len(pieces) != 3:
        raise ValueError(f"unexpected result directory: {directory}")
    model_slug, watermark, dataset = pieces
    row = _original_metric(metrics_path)
    model_name = str(row["model_name"])
    if MODEL_SLUG_TO_NAME.get(model_slug) != model_name:
        raise ValueError(f"model mismatch in {metrics_path}: {model_slug} / {model_name}")
    generate_path = directory / "generate.jsonl"
    expected = EXPECTED_TASKS[dataset]
    actual = sum(1 for _ in iter_jsonl(generate_path))
    if actual != expected:
        raise ValueError(f"unexpected row count in {generate_path}: {actual} != {expected}")
    return {
        "key": f"{model_slug}--{watermark}--{dataset}--{directory.name}",
        "model_slug": model_slug,
        "model_name": model_name,
        "watermark": watermark,
        "dataset": dataset,
        "config_id": directory.name,
        "generate_path": relative_to_repo(generate_path),
        "metrics_path": relative_to_repo(metrics_path),
        "temperature": float(row["temperature"]),
        "delta": None if row.get("delta") is None else float(row["delta"]),
        "gamma": None if row.get("gamma") is None else float(row["gamma"]),
        "entropy_threshold": (
            None
            if row.get("entropy_threshold") is None
            else float(row["entropy_threshold"])
        ),
        "ngram_len": int(row["ngram_len"]),
        "original_auroc": float(row["auroc"]),
        "original_pass1": float(row["pass1"]),
    }


def discover_candidates() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    configs: dict[str, dict[str, Any]] = {}
    pattern = "*--*--*_py/*/metrics.jsonl"
    for metrics_path in sorted(RESULT_ROOT.glob(pattern)):
        pieces = metrics_path.parent.parent.name.split("--")
        if len(pieces) != 3:
            continue
        _, watermark, dataset = pieces
        if watermark not in SCHEMES or dataset not in EXPECTED_TASKS:
            continue
        config = _config_from_metrics(metrics_path)
        configs[config["key"]] = config
        for record in iter_jsonl(resolve_repo_path(config["generate_path"])):
            z_score = record.get("z_score")
            if record.get("passed") is not True or z_score is None or float(z_score) <= 4.0:
                continue
            record_id = record.get("id")
            task_name = record.get("task_name")
            solution = record.get("solution")
            if not all(isinstance(value, str) and value for value in (record_id, task_name, solution)):
                raise ValueError(f"invalid selected record in {config['generate_path']}")
            candidate_key = f"{config['key']}::{task_name}"
            candidates.append(
                {
                    "candidate_key": candidate_key,
                    "record_id": record_id,
                    "task_name": task_name,
                    "config_key": config["key"],
                    "model_slug": config["model_slug"],
                    "watermark": config["watermark"],
                    "dataset": config["dataset"],
                    "generate_path": config["generate_path"],
                    "original_z_score": float(z_score),
                    "solution_sha256": hashlib.sha256(solution.encode("utf-8")).hexdigest(),
                }
            )
    candidates.sort(key=lambda row: str(row["candidate_key"]))
    if len(candidates) != 1620:
        raise ValueError(f"candidate universe changed: {len(candidates)} != 1620")
    if len({row["candidate_key"] for row in candidates}) != len(candidates):
        raise ValueError("candidate keys are not unique")
    if len({row["record_id"] for row in candidates}) != len(candidates):
        raise ValueError("record ids are not unique")
    if len({row["solution_sha256"] for row in candidates}) != len(candidates):
        raise ValueError("selected solutions are not unique")
    return candidates, configs


def build_manifest(run_id: str = RUN_ID) -> dict[str, Any]:
    candidates, all_configs = discover_candidates()
    rng = random.Random(GLOBAL_SEED)
    sampled = rng.sample(candidates, SAMPLE_SIZE)
    sampled.sort(key=lambda row: str(row["candidate_key"]))
    config_keys = sorted({str(row["config_key"]) for row in sampled})
    configs = [all_configs[key] for key in config_keys]
    trajectory_count = SAMPLE_SIZE * TRAJECTORIES_PER_SEED
    shards = [
        {
            "index": index,
            "task_start": start,
            "task_stop": min(start + TASKS_PER_SHARD, trajectory_count),
        }
        for index, start in enumerate(range(0, trajectory_count, TASKS_PER_SHARD))
    ]
    input_paths = sorted(
        {
            resolve_repo_path(config["generate_path"])
            for config in configs
        }
        | {
            resolve_repo_path(config["metrics_path"])
            for config in configs
        }
    )
    source_paths = sorted((NEW_PROTOTYPE_ROOT / "rw_obfuscator").rglob("*.py"))
    scheme_seeds = Counter(str(row["watermark"]) for row in sampled)
    model_seeds = Counter(str(row["model_slug"]) for row in sampled)
    dataset_seeds = Counter(str(row["dataset"]) for row in sampled)
    return {
        "schema_version": 1,
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection": {
            "universe": "all real Python watermark configurations",
            "saved_original_passed": True,
            "original_z_score_strictly_above": 4.0,
            "candidate_count": len(candidates),
            "sample_without_replacement": True,
            "sample_seed": GLOBAL_SEED,
        },
        "walk": {
            "rule_profile": "full",
            "steps": STEPS,
            "trajectories_per_seed": TRAJECTORIES_PER_SEED,
            "global_seed": GLOBAL_SEED,
            "identity_is_uniform_concrete_action": True,
            "retry_or_resample": False,
            "retain_duplicate_endpoints": True,
            "save_every_program": True,
        },
        "test": {
            "execution_hash_seed": GLOBAL_SEED,
            "paper_compatible": "scipy.stats.anderson(dist='norm')",
            "primary_significance_level_percent": 5.0,
            "secondary_significance_level_percent": 15.0,
        },
        "candidates": sampled,
        "configs": configs,
        "shards": shards,
        "counts": {
            "candidate_universe": len(candidates),
            "sampled_seeds": len(sampled),
            "sampled_seeds_by_scheme": dict(sorted(scheme_seeds.items())),
            "sampled_seeds_by_model": dict(sorted(model_seeds.items())),
            "sampled_seeds_by_dataset": dict(sorted(dataset_seeds.items())),
            "sampled_configs": len(configs),
            "trajectories": trajectory_count,
            "transitions": trajectory_count * STEPS,
            "saved_programs": trajectory_count * (STEPS + 1),
            "transform_shards": len(shards),
        },
        "input_files": {
            relative_to_repo(path): {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in input_paths
        },
        "obfuscator_sources": {
            relative_to_repo(path): sha256_file(path) for path in source_paths
        },
        "environment": {"python": sys.version.split()[0]},
    }


def build_rejected_followup_manifest(
    run_id: str,
    source_run_id: str,
    *,
    steps: int = FOLLOWUP_STEPS,
    selection: str = "paper-reject-5",
    global_seed: int | None = None,
    verify_source_prefix: bool = True,
) -> dict[str, Any]:
    """Build a longer-walk run from selected spaces in a completed source run."""

    if steps <= 0:
        raise ValueError("steps must be positive")
    if selection not in {"paper-reject-5", "all"}:
        raise ValueError(f"unsupported follow-up selection: {selection!r}")
    source_root = run_root(source_run_id)
    source_manifest_path = source_root / "manifest.json"
    source_spaces_path = source_root / "spaces.jsonl"
    if not source_manifest_path.is_file() or not source_spaces_path.is_file():
        raise FileNotFoundError(
            f"source run needs manifest.json and spaces.jsonl: {source_root}"
        )
    source_manifest = load_manifest(source_run_id)
    if selection == "paper-reject-5":
        selected_rows = [
            row
            for row in iter_jsonl(source_spaces_path)
            if row.get("paper_ad_accept_5") is False
        ]
        selected_keys = {str(row["candidate_key"]) for row in selected_rows}
        if not selected_keys:
            raise ValueError(
                f"source run has no paper-AD rejections at 5%: {source_run_id}"
            )
    else:
        selected_keys = {
            str(candidate["candidate_key"])
            for candidate in source_manifest["candidates"]
        }
    candidates = [
        copy.deepcopy(row)
        for row in source_manifest["candidates"]
        if str(row["candidate_key"]) in selected_keys
    ]
    candidates.sort(key=lambda row: str(row["candidate_key"]))
    if {str(row["candidate_key"]) for row in candidates} != selected_keys:
        raise ValueError("source spaces and manifest candidate sets do not match")
    config_keys = {str(row["config_key"]) for row in candidates}
    configs = [
        copy.deepcopy(row)
        for row in source_manifest["configs"]
        if str(row["key"]) in config_keys
    ]
    configs.sort(key=lambda row: str(row["key"]))

    trajectories_per_seed = int(source_manifest["walk"]["trajectories_per_seed"])
    source_global_seed = int(source_manifest["walk"]["global_seed"])
    selected_global_seed = (
        source_global_seed if global_seed is None else int(global_seed)
    )
    trajectory_count = len(candidates) * trajectories_per_seed
    # One trajectory per shard lets the small follow-up cohort still use all
    # allocated transform workers.
    shards = [
        {
            "index": index,
            "task_start": start,
            "task_stop": start + 1,
        }
        for index, start in enumerate(range(trajectory_count))
    ]
    input_paths = sorted(
        {
            resolve_repo_path(str(config["generate_path"]))
            for config in configs
        }
        | {
            resolve_repo_path(str(config["metrics_path"]))
            for config in configs
        }
    )
    source_paths = sorted((NEW_PROTOTYPE_ROOT / "rw_obfuscator").rglob("*.py"))
    scheme_seeds = Counter(str(row["watermark"]) for row in candidates)
    model_seeds = Counter(str(row["model_slug"]) for row in candidates)
    dataset_seeds = Counter(str(row["dataset"]) for row in candidates)
    return {
        "schema_version": 1,
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection": {
            "source_run_id": source_run_id,
            "selection_mode": selection,
            "source_test": "scipy.stats.anderson(dist='norm')",
            "source_significance_level_percent": 5.0,
            "source_hypothesis_decision": (
                "reject normality null"
                if selection == "paper-reject-5"
                else "all source spaces"
            ),
            "verify_source_prefix": verify_source_prefix,
            "source_spaces": len(source_manifest["candidates"]),
            "selected_spaces": len(candidates),
        },
        "walk": {
            "rule_profile": str(source_manifest["walk"]["rule_profile"]),
            "steps": steps,
            "trajectories_per_seed": trajectories_per_seed,
            "global_seed": selected_global_seed,
            "identity_is_uniform_concrete_action": True,
            "retry_or_resample": False,
            "retain_duplicate_endpoints": True,
            "save_every_program": True,
        },
        "test": {
            "execution_hash_seed": int(
                source_manifest["test"].get(
                    "execution_hash_seed", source_global_seed
                )
            ),
            "paper_compatible": "scipy.stats.anderson(dist='norm')",
            "primary_significance_level_percent": 5.0,
            "secondary_significance_level_percent": 15.0,
        },
        "candidates": candidates,
        "configs": configs,
        "shards": shards,
        "counts": {
            "source_spaces": len(source_manifest["candidates"]),
            "sampled_seeds": len(candidates),
            "sampled_seeds_by_scheme": dict(sorted(scheme_seeds.items())),
            "sampled_seeds_by_model": dict(sorted(model_seeds.items())),
            "sampled_seeds_by_dataset": dict(sorted(dataset_seeds.items())),
            "sampled_configs": len(configs),
            "trajectories": trajectory_count,
            "transitions": trajectory_count * steps,
            "saved_programs": trajectory_count * (steps + 1),
            "transform_shards": len(shards),
        },
        "source_artifacts": {
            relative_to_repo(source_manifest_path): sha256_file(source_manifest_path),
            relative_to_repo(source_spaces_path): sha256_file(source_spaces_path),
        },
        "input_files": {
            relative_to_repo(path): {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in input_paths
        },
        "obfuscator_sources": {
            relative_to_repo(path): sha256_file(path) for path in source_paths
        },
        "environment": {"python": sys.version.split()[0]},
    }


def build_useful_parseable_sample_manifest(
    run_id: str,
    *,
    source_run_id: str = USEFUL_SOURCE_RUN_ID,
) -> dict[str, Any]:
    """Sample parseable programs from the frozen 43 useful configurations."""

    source_root = (
        NEW_PROTOTYPE_ROOT / "data" / "watermark_attack" / source_run_id
    ).resolve()
    source_root.relative_to((NEW_PROTOTYPE_ROOT / "data" / "watermark_attack").resolve())
    source_manifest_path = source_root / "manifest.json"
    if not source_manifest_path.is_file():
        raise FileNotFoundError(source_manifest_path)
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("counts", {}).get("configs") != 43:
        raise ValueError("useful source run no longer contains 43 configurations")
    expected_selection = {
        "negative_distribution": "standard_normal",
        "original_auroc_strictly_above": 0.8,
        "original_pass1_retention_strictly_above": 0.8,
    }
    if source_manifest.get("selection") != expected_selection:
        raise ValueError("useful source selection contract changed")

    all_records = 0
    unparseable = 0
    candidates: list[dict[str, Any]] = []
    for config in source_manifest["configs"]:
        for record in iter_jsonl(resolve_repo_path(str(config["generate_path"]))):
            all_records += 1
            solution = record.get("solution")
            record_id = record.get("id")
            task_name = record.get("task_name")
            z_score = record.get("z_score")
            if not all(
                isinstance(value, str) and value
                for value in (solution, record_id, task_name)
            ) or z_score is None:
                raise ValueError(f"invalid generation record in {config['generate_path']}")
            try:
                ast.parse(solution)
            except (SyntaxError, ValueError):
                unparseable += 1
                continue
            candidates.append(
                {
                    "candidate_key": f"{config['key']}::{task_name}",
                    "record_id": record_id,
                    "task_name": task_name,
                    "config_key": config["key"],
                    "model_slug": config["model_slug"],
                    "watermark": config["watermark"],
                    "dataset": config["dataset"],
                    "generate_path": config["generate_path"],
                    "original_passed": record.get("passed") is True,
                    "original_z_score": float(z_score),
                    "solution_sha256": hashlib.sha256(solution.encode("utf-8")).hexdigest(),
                }
            )
    candidates.sort(key=lambda row: str(row["candidate_key"]))
    if all_records != 11_118 or len(candidates) != 10_877 or unparseable != 241:
        raise ValueError(
            "useful candidate universe changed: "
            f"all={all_records} parseable={len(candidates)} unparseable={unparseable}"
        )
    if len({str(row["candidate_key"]) for row in candidates}) != len(candidates):
        raise ValueError("candidate keys are not unique")
    if len({str(row["record_id"]) for row in candidates}) != len(candidates):
        raise ValueError("record ids are not unique")

    sampled = random.Random(GLOBAL_SEED).sample(candidates, SAMPLE_SIZE)
    sampled.sort(key=lambda row: str(row["candidate_key"]))
    config_keys = {str(row["config_key"]) for row in sampled}
    configs = [
        copy.deepcopy(config)
        for config in source_manifest["configs"]
        if str(config["key"]) in config_keys
    ]
    configs.sort(key=lambda row: str(row["key"]))
    trajectory_count = SAMPLE_SIZE * TRAJECTORIES_PER_SEED
    shards = [
        {
            "index": index,
            "task_start": start,
            "task_stop": min(start + TASKS_PER_SHARD, trajectory_count),
        }
        for index, start in enumerate(range(0, trajectory_count, TASKS_PER_SHARD))
    ]
    input_paths = sorted(
        {
            resolve_repo_path(str(config["generate_path"]))
            for config in configs
        }
        | {
            resolve_repo_path(str(config["metrics_path"]))
            for config in configs
        }
    )
    source_paths = sorted((NEW_PROTOTYPE_ROOT / "rw_obfuscator").rglob("*.py"))
    scheme_seeds = Counter(str(row["watermark"]) for row in sampled)
    model_seeds = Counter(str(row["model_slug"]) for row in sampled)
    dataset_seeds = Counter(str(row["dataset"]) for row in sampled)
    passed_seeds = Counter(
        "passed" if bool(row["original_passed"]) else "failed" for row in sampled
    )
    return {
        "schema_version": 1,
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection": {
            "source_watermark_attack_run_id": source_run_id,
            "source_configs": 43,
            "source_original_auroc_strictly_above": 0.8,
            "source_original_pass1_retention_strictly_above": 0.8,
            "source_records": all_records,
            "require_original_passed": False,
            "require_python_parseable": True,
            "candidate_count": len(candidates),
            "sample_without_replacement": True,
            "sample_seed": GLOBAL_SEED,
        },
        "walk": {
            "rule_profile": "full",
            "steps": STEPS,
            "trajectories_per_seed": TRAJECTORIES_PER_SEED,
            "global_seed": GLOBAL_SEED,
            "identity_is_uniform_concrete_action": True,
            "retry_or_resample": False,
            "retain_duplicate_endpoints": True,
            "save_every_program": True,
        },
        "test": {
            "execution_hash_seed": GLOBAL_SEED,
            "require_baseline_passed": False,
            "require_same_test_outcome": True,
            "paper_compatible": "scipy.stats.anderson(dist='norm')",
            "primary_significance_level_percent": 5.0,
            "secondary_significance_level_percent": 15.0,
        },
        "candidates": sampled,
        "configs": configs,
        "shards": shards,
        "counts": {
            "source_records": all_records,
            "parseable_candidates": len(candidates),
            "unparseable_excluded": unparseable,
            "sampled_seeds": len(sampled),
            "sampled_seeds_by_original_test": dict(sorted(passed_seeds.items())),
            "sampled_seeds_by_scheme": dict(sorted(scheme_seeds.items())),
            "sampled_seeds_by_model": dict(sorted(model_seeds.items())),
            "sampled_seeds_by_dataset": dict(sorted(dataset_seeds.items())),
            "sampled_configs": len(configs),
            "trajectories": trajectory_count,
            "transitions": trajectory_count * STEPS,
            "saved_programs": trajectory_count * (STEPS + 1),
            "transform_shards": len(shards),
        },
        "source_artifacts": {
            relative_to_repo(source_manifest_path): sha256_file(source_manifest_path),
        },
        "input_files": {
            relative_to_repo(path): {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in input_paths
        },
        "obfuscator_sources": {
            relative_to_repo(path): sha256_file(path) for path in source_paths
        },
        "environment": {"python": sys.version.split()[0]},
    }


def load_manifest(run_id: str) -> dict[str, Any]:
    path = run_root(run_id) / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def config_map(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["key"]): dict(row) for row in manifest["configs"]}


@lru_cache(maxsize=4)
def _records_by_id(generate_path: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(resolve_repo_path(generate_path)):
        record_id = row.get("id")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError(f"record without a valid id in {generate_path}")
        if record_id in result:
            raise ValueError(f"duplicate record id {record_id!r} in {generate_path}")
        result[record_id] = row
    return result


def record_for_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    generate_path = str(candidate["generate_path"])
    record_id = str(candidate["record_id"])
    record = _records_by_id(generate_path).get(record_id)
    if record is None:
        raise ValueError(f"expected one source record for {candidate['candidate_key']}")
    return record


def trajectory_identity(candidate: Mapping[str, Any], trajectory_index: int) -> str:
    return f"{candidate['candidate_key']}::trajectory-{trajectory_index:02d}"


def _transform_task(
    run_id: str,
    task_index: int,
    manifest: Mapping[str, Any] | None = None,
    source_record: Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
    precomputed_baseline_execution: Any | None = None,
) -> dict[str, Any]:
    # Imported lazily so manifest/statistics commands do not require LibCST.
    from experiment.transform import transform_record

    manifest = dict(manifest or load_manifest(run_id))
    trajectories_per_seed = int(manifest["walk"]["trajectories_per_seed"])
    candidate_index, trajectory_index = divmod(task_index, trajectories_per_seed)
    candidate = dict(manifest["candidates"][candidate_index])
    config = dict(config or config_map(manifest)[str(candidate["config_key"])])
    source_record = dict(source_record or record_for_candidate(candidate))
    original_record_id = str(source_record["id"])
    trajectory_id = trajectory_identity(candidate, trajectory_index)
    trajectory_record = dict(source_record)
    trajectory_record["id"] = trajectory_id
    case_global_seed = int(
        candidate.get("random_seed", manifest["walk"]["global_seed"])
    )
    result = transform_record(
        trajectory_record,
        config=config,
        steps=int(manifest["walk"]["steps"]),
        global_seed=case_global_seed,
        timeout=TRANSFORM_TIMEOUT_SECONDS,
        memory_mb=TRANSFORM_MEMORY_MB,
        rule_profile=str(manifest["walk"].get("rule_profile", "full")),
        execution_hash_seed=int(
            manifest["test"].get(
                "execution_hash_seed", manifest["walk"]["global_seed"]
            )
        ),
        precomputed_baseline_execution=precomputed_baseline_execution,
    )
    result.update(
        {
            "candidate_key": candidate["candidate_key"],
            "trajectory_index": trajectory_index,
            "trajectory_id": trajectory_id,
            "source_record_id": original_record_id,
            "original_z_score": candidate["original_z_score"],
        }
    )
    return result


def transform_shard(run_id: str, shard_index: int) -> Path:
    from rw_obfuscator.corpus import execute_case

    manifest = load_manifest(run_id)
    shard = manifest["shards"][shard_index]
    output = run_root(run_id) / "transforms" / f"part-{shard_index:04d}.jsonl.gz"
    if output.is_file():
        return output
    trajectories_per_seed = int(manifest["walk"]["trajectories_per_seed"])
    configs = config_map(manifest)
    source_cache: dict[int, dict[str, Any]] = {}
    baseline_cache: dict[int, Any] = {}
    rows: list[dict[str, Any]] = []
    for task_index in range(int(shard["task_start"]), int(shard["task_stop"])):
        candidate_index, _ = divmod(task_index, trajectories_per_seed)
        candidate = dict(manifest["candidates"][candidate_index])
        if candidate_index not in source_cache:
            source_cache[candidate_index] = record_for_candidate(candidate)
        source_record = source_cache[candidate_index]
        if candidate_index not in baseline_cache:
            source = source_record.get("solution")
            test = source_record.get("test")
            baseline_cache[candidate_index] = (
                execute_case(
                    source,
                    test,
                    timeout=TRANSFORM_TIMEOUT_SECONDS,
                    memory_mb=TRANSFORM_MEMORY_MB,
                    hash_seed=int(
                        manifest["test"].get(
                            "execution_hash_seed", manifest["walk"]["global_seed"]
                        )
                    ),
                )
                if isinstance(source, str) and isinstance(test, str)
                else None
            )
        rows.append(
            _transform_task(
                run_id,
                task_index,
                manifest,
                source_record=source_record,
                config=configs[str(candidate["config_key"])],
                precomputed_baseline_execution=baseline_cache[candidate_index],
            )
        )
    atomic_jsonl(output, rows)
    return output


def command_transform_array(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.run_id)
    shard_indices = [
        index
        for index in range(len(manifest["shards"]))
        if index % args.array_count == args.array_index
    ]
    failures: list[tuple[int, BaseException]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(transform_shard, args.run_id, index): index
            for index in shard_indices
        }
        for future in as_completed(futures):
            index = futures[future]
            try:
                print(f"completed shard {index}: {future.result()}", flush=True)
            except BaseException as error:
                failures.append((index, error))
                print(f"failed shard {index}: {type(error).__name__}: {error}", flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} transform shards failed")


def iter_transform_rows(run_id: str) -> Iterator[dict[str, Any]]:
    for path in sorted((run_root(run_id) / "transforms").glob("part-*.jsonl.gz")):
        yield from iter_jsonl(path)


def _write_endpoint_index(
    run_id: str,
    manifest: Mapping[str, Any],
    rows: Iterable[Mapping[str, Any]],
) -> int:
    by_config: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        programs = row.get("programs")
        if not isinstance(programs, list) or not programs or not isinstance(
            programs[-1], str
        ):
            raise ValueError(f"invalid endpoint program for {row.get('trajectory_id')}")
        baseline = row.get("baseline", {})
        baseline_execution = baseline.get("execution", {})
        final_execution = row.get("final", {}).get("execution", {})
        endpoint = {
            "schema_version": 1,
            "config_key": row["config_key"],
            "candidate_key": row["candidate_key"],
            "trajectory_index": row["trajectory_index"],
            "trajectory_id": row["trajectory_id"],
            "status": row["status"],
            "detection_g4d": row["detection_g4d"],
            "endpoint_program": programs[-1],
            "saved_original_passed": baseline.get(
                "saved_passed", baseline.get("passed")
            ),
            "baseline_execution_passed": baseline_execution.get("passed"),
            "final_execution_passed": final_execution.get("passed"),
        }
        by_config[str(row["config_key"])].append(endpoint)

    trajectories_per_seed = int(manifest["walk"]["trajectories_per_seed"])
    candidates_per_config = Counter(
        str(candidate["config_key"]) for candidate in manifest["candidates"]
    )
    if set(by_config) != set(candidates_per_config):
        raise ValueError("endpoint index config set differs from manifest")
    output_root = run_root(run_id) / "endpoints"
    for config_key in sorted(by_config):
        config_rows = sorted(
            by_config[config_key],
            key=lambda row: (
                str(row["candidate_key"]),
                int(row["trajectory_index"]),
            ),
        )
        expected = candidates_per_config[config_key] * trajectories_per_seed
        if len(config_rows) != expected:
            raise ValueError(
                f"endpoint index {config_key}: {len(config_rows)} != {expected}"
            )
        output = output_root / f"{config_key}.jsonl.gz"
        atomic_jsonl(output, config_rows, overwrite=output.exists())
    return len(by_config)


def command_validate(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.run_id)
    steps = int(manifest["walk"]["steps"])
    paths = sorted((run_root(args.run_id) / "transforms").glob("part-*.jsonl.gz"))
    expected_shards = int(manifest["counts"]["transform_shards"])
    if len(paths) != expected_shards:
        raise RuntimeError(f"transform shards: {len(paths)} != {expected_shards}")
    rows = list(iter_transform_rows(args.run_id))
    expected_rows = int(manifest["counts"]["trajectories"])
    if len(rows) != expected_rows:
        raise RuntimeError(f"transform rows: {len(rows)} != {expected_rows}")
    ids = [str(row.get("trajectory_id")) for row in rows]
    if len(set(ids)) != len(ids):
        raise RuntimeError("duplicate trajectory ids")
    source_rows: dict[str, dict[str, Any]] = {}
    source_run_id = manifest.get("selection", {}).get("source_run_id")
    source_steps = 0
    if isinstance(source_run_id, str) and bool(
        manifest.get("selection", {}).get("verify_source_prefix", True)
    ):
        source_manifest = load_manifest(source_run_id)
        source_steps = int(source_manifest["walk"]["steps"])
        selected_candidates = {
            str(candidate["candidate_key"]) for candidate in manifest["candidates"]
        }
        source_rows = {
            str(row["trajectory_id"]): row
            for row in iter_transform_rows(source_run_id)
            if str(row.get("candidate_key")) in selected_candidates
        }
        if set(source_rows) != set(ids):
            raise RuntimeError("source trajectory set does not match follow-up")
    failures: list[str] = []
    require_baseline_passed = bool(
        manifest["test"].get("require_baseline_passed", True)
    )
    require_same_test_outcome = bool(
        manifest["test"].get("require_same_test_outcome", False)
    )
    for row in rows:
        reasons = []
        if row.get("status") != "ok":
            reasons.append(f"status={row.get('status')} error={row.get('error')}")
        if len(row.get("programs", ())) != steps + 1:
            reasons.append(f"programs={len(row.get('programs', ()))}")
        if len(row.get("trace", ())) != steps:
            reasons.append(f"trace={len(row.get('trace', ()))}")
        baseline_passed = row.get("baseline", {}).get("execution", {}).get("passed")
        final_passed = row.get("final", {}).get("execution", {}).get("passed")
        if require_baseline_passed and baseline_passed is not True:
            reasons.append("baseline did not pass")
        if require_baseline_passed and final_passed is not True:
            reasons.append("final did not pass")
        if require_same_test_outcome and baseline_passed != final_passed:
            reasons.append(
                f"test outcome changed: baseline={baseline_passed} final={final_passed}"
            )
        if not isinstance(row.get("detection_g4d"), str):
            reasons.append("missing detection_g4d")
        if source_rows:
            source = source_rows[str(row["trajectory_id"])]
            if row.get("programs", ())[: source_steps + 1] != source.get("programs"):
                reasons.append("first source_steps+1 programs differ from source run")
            if row.get("trace", ())[:source_steps] != source.get("trace"):
                reasons.append("first source_steps trace entries differ from source run")
        if reasons:
            failures.append(f"{row.get('trajectory_id')}: " + "; ".join(reasons))
    if failures:
        raise RuntimeError(
            f"{len(failures)} strict trajectory failures\n" + "\n".join(failures[:20])
        )
    endpoint_configs = _write_endpoint_index(args.run_id, manifest, rows)
    print(
        json.dumps(
            {
                "rows": len(rows),
                "strict_failures": 0,
                "endpoint_configs": endpoint_configs,
            },
            indent=2,
        )
    )


def assigned_config_keys(
    manifest: Mapping[str, Any],
    *,
    scheme: str,
    model_slug: str | None,
    shard_index: int,
    shard_count: int,
) -> list[str]:
    if shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise ValueError("invalid score shard index/count")
    configs = config_map(manifest)
    keys = sorted(
        key
        for key, config in configs.items()
        if config["watermark"] == scheme
        and (model_slug is None or config["model_slug"] == model_slug)
    )
    candidate_counts = Counter(
        str(candidate["config_key"]) for candidate in manifest["candidates"]
    )
    trajectories_per_seed = int(manifest["walk"]["trajectories_per_seed"])
    weights = {
        key: candidate_counts[key] * trajectories_per_seed for key in keys
    }
    assignments: list[list[str]] = [[] for _ in range(shard_count)]
    loads = [0] * shard_count
    for key in sorted(keys, key=lambda value: (-weights[value], value)):
        destination = min(range(shard_count), key=lambda index: (loads[index], index))
        assignments[destination].append(key)
        loads[destination] += weights[key]
    return sorted(assignments[shard_index])


def _transforms_by_config(
    run_id: str,
    selected_configs: set[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in iter_transform_rows(run_id):
        config_key = str(row["config_key"])
        if selected_configs is None or config_key in selected_configs:
            result[config_key].append(row)
    return result


def _scoring_inputs_by_config(
    run_id: str, selected_configs: set[str]
) -> dict[str, list[dict[str, Any]]]:
    endpoint_root = run_root(run_id) / "endpoints"
    endpoint_paths = {
        key: endpoint_root / f"{key}.jsonl.gz" for key in selected_configs
    }
    if all(path.is_file() for path in endpoint_paths.values()):
        result: dict[str, list[dict[str, Any]]] = {}
        for key in sorted(endpoint_paths):
            rows = list(iter_jsonl(endpoint_paths[key]))
            if any(str(row.get("config_key")) != key for row in rows):
                raise ValueError(f"endpoint index contains a foreign config: {key}")
            result[key] = rows
        return result
    return _transforms_by_config(run_id, selected_configs)


def command_score_group(args: argparse.Namespace) -> None:
    from experiment.detectors import make_scorer

    manifest = load_manifest(args.run_id)
    configs = config_map(manifest)
    candidates = {str(row["candidate_key"]): row for row in manifest["candidates"]}
    assigned = assigned_config_keys(
        manifest,
        scheme=args.scheme,
        model_slug=args.model_slug,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )
    expected_candidates_by_key: dict[str, set[str]] = {
        key: {
            candidate_key
            for candidate_key, candidate in candidates.items()
            if str(candidate["config_key"]) == key
        }
        for key in assigned
    }
    pending: list[str] = []
    for key in assigned:
        output = run_root(args.run_id) / "scores" / f"{key}.jsonl.gz"
        expected_candidates = expected_candidates_by_key[key]
        expected_rows = len(expected_candidates) * int(
            manifest["walk"]["trajectories_per_seed"]
        )
        if output.is_file():
            existing = list(iter_jsonl(output))
            existing_candidates = {
                str(row.get("candidate_key")) for row in existing
            }
            if (
                len(existing) == expected_rows
                and existing_candidates == expected_candidates
                and all(row.get("status") == "ok" for row in existing)
            ):
                print(f"skip complete score file {key}: {len(existing)}", flush=True)
                continue
        pending.append(key)

    if not pending:
        return
    transforms = _scoring_inputs_by_config(args.run_id, set(pending))
    for key in pending:
        config = configs[key]
        output = run_root(args.run_id) / "scores" / f"{key}.jsonl.gz"
        scorer = make_scorer(config, synthid_device=args.synthid_device)
        rows: list[dict[str, Any]] = []
        source_cache: dict[str, dict[str, Any]] = {}
        checked_candidates: set[str] = set()
        for transformed in sorted(
            transforms.get(key, ()),
            key=lambda row: (str(row["candidate_key"]), int(row["trajectory_index"])),
        ):
            candidate_key = str(transformed["candidate_key"])
            if candidate_key not in source_cache:
                source_cache[candidate_key] = record_for_candidate(candidates[candidate_key])
            source = source_cache[candidate_key]
            if candidate_key not in checked_candidates:
                baseline = scorer.score(source, str(source["g4d"]))
                if not math.isclose(
                    float(baseline["z_score"]),
                    float(source["z_score"]),
                    rel_tol=0.0,
                    abs_tol=1e-10,
                ):
                    raise RuntimeError(
                        f"baseline z mismatch for {candidate_key}: "
                        f"{baseline['z_score']} != {source['z_score']}"
                    )
                checked_candidates.add(candidate_key)
            score = scorer.score(source, str(transformed["detection_g4d"]))
            endpoint_program = transformed.get("endpoint_program")
            if endpoint_program is None:
                programs = transformed.get("programs")
                endpoint_program = (
                    programs[-1] if isinstance(programs, list) and programs else None
                )
            if not isinstance(endpoint_program, str):
                raise ValueError(f"missing endpoint program for {transformed['trajectory_id']}")
            baseline_passed = transformed.get("baseline_execution_passed")
            if "baseline_execution_passed" not in transformed:
                baseline_passed = (
                    transformed.get("baseline", {}).get("execution", {}).get("passed")
                )
            final_passed = transformed.get("final_execution_passed")
            if "final_execution_passed" not in transformed:
                final_passed = (
                    transformed.get("final", {}).get("execution", {}).get("passed")
                )
            rows.append(
                {
                    "schema_version": 1,
                    "config_key": key,
                    "candidate_key": candidate_key,
                    "trajectory_index": transformed["trajectory_index"],
                    "trajectory_id": transformed["trajectory_id"],
                    "watermark": config["watermark"],
                    "model_slug": config["model_slug"],
                    "dataset": config["dataset"],
                    "status": "ok",
                    "saved_original_passed": transformed.get(
                        "saved_original_passed", source.get("passed")
                    ),
                    "baseline_execution_passed": baseline_passed,
                    "final_execution_passed": final_passed,
                    "endpoint_program": endpoint_program,
                    "score": score,
                }
            )
        atomic_jsonl(output, rows, overwrite=output.exists())
        print(f"scored {key}: {len(rows)}", flush=True)


def fixed_standard_normal_ad(values: Iterable[float]) -> float:
    import numpy as np
    from scipy import stats

    ordered = np.sort(np.asarray(list(values), dtype=float))
    n = ordered.size
    if n == 0:
        raise ValueError("empty sample")
    cdf = np.clip(stats.norm.cdf(ordered), 1e-15, 1.0 - 1e-15)
    weights = 2 * np.arange(1, n + 1) - 1
    return float(-n - np.sum(weights * (np.log(cdf) + np.log(1.0 - cdf[::-1]))) / n)


def fixed_ad_null(draws: int, sample_size: int, seed: int) -> np.ndarray:
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(seed)
    output = np.empty(draws, dtype=float)
    batch_size = 5000
    weights = 2 * np.arange(1, sample_size + 1) - 1
    for start in range(0, draws, batch_size):
        stop = min(start + batch_size, draws)
        samples = np.sort(rng.standard_normal((stop - start, sample_size)), axis=1)
        cdf = np.clip(stats.norm.cdf(samples), 1e-15, 1.0 - 1e-15)
        output[start:stop] = -sample_size - np.sum(
            weights * (np.log(cdf) + np.log(1.0 - cdf[:, ::-1])), axis=1
        ) / sample_size
    return output


def _anderson_critical(result: Any, level: float) -> float:
    for significance, critical in zip(result.significance_level, result.critical_values):
        if math.isclose(float(significance), level, rel_tol=0.0, abs_tol=1e-12):
            return float(critical)
    raise ValueError(f"Anderson result lacks significance level {level}")


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def command_aggregate(args: argparse.Namespace) -> None:
    import numpy as np
    from scipy import stats

    manifest = load_manifest(args.run_id)
    trajectories_per_seed = int(manifest["walk"]["trajectories_per_seed"])
    include_fixed_tests = "fixed_standard_normal_monte_carlo_draws" in manifest["test"]
    score_paths = sorted((run_root(args.run_id) / "scores").glob("*.jsonl.gz"))
    score_rows = [row for path in score_paths for row in iter_jsonl(path)]
    expected = int(manifest["counts"]["trajectories"])
    if len(score_rows) != expected:
        raise RuntimeError(f"score rows: {len(score_rows)} != {expected}")
    if any(row.get("status") != "ok" or not row.get("score") for row in score_rows):
        raise RuntimeError("non-ok score row")
    by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in score_rows:
        by_candidate[str(row["candidate_key"])].append(row)
    if set(by_candidate) != {str(row["candidate_key"]) for row in manifest["candidates"]}:
        raise RuntimeError("scored candidate set mismatch")
    if include_fixed_tests:
        draws = int(manifest["test"]["fixed_standard_normal_monte_carlo_draws"])
        fixed_seed = int(manifest["test"]["fixed_standard_normal_seed"])
        null = fixed_ad_null(draws, trajectories_per_seed, fixed_seed)
        fixed_critical_5 = float(np.quantile(null, 0.95))
        fixed_critical_15 = float(np.quantile(null, 0.85))
    candidate_map = {str(row["candidate_key"]): row for row in manifest["candidates"]}
    space_rows: list[dict[str, Any]] = []
    for key in sorted(by_candidate):
        rows = sorted(by_candidate[key], key=lambda row: int(row["trajectory_index"]))
        if len(rows) != trajectories_per_seed:
            raise RuntimeError(f"{key}: {len(rows)} scores")
        values = [float(row["score"]["z_score"]) for row in rows]
        paper = stats.anderson(values, dist="norm")
        paper_critical_5 = _anderson_critical(paper, 5.0)
        paper_critical_15 = _anderson_critical(paper, 15.0)
        candidate = candidate_map[key]
        space_row = {
            "candidate_key": key,
            "config_key": candidate["config_key"],
            "task_name": candidate["task_name"],
            "watermark": candidate["watermark"],
            "model_slug": candidate["model_slug"],
            "dataset": candidate["dataset"],
            "original_z_score": candidate["original_z_score"],
            "samples": len(values),
            "sample_mean": float(np.mean(values)),
            "sample_std": float(np.std(values, ddof=1)),
            "sample_min": min(values),
            "sample_max": max(values),
            "paper_ad_statistic": float(paper.statistic),
            "paper_ad_critical_5": paper_critical_5,
            "paper_ad_accept_5": bool(float(paper.statistic) <= paper_critical_5),
            "paper_ad_critical_15": paper_critical_15,
            "paper_ad_accept_15": bool(float(paper.statistic) <= paper_critical_15),
        }
        if include_fixed_tests:
            fixed = fixed_standard_normal_ad(values)
            fixed_p = float((1 + np.count_nonzero(null >= fixed)) / (draws + 1))
            ks = stats.kstest(values, stats.norm.cdf)
            space_row.update(
                {
                    "fixed_ad_statistic": fixed,
                    "fixed_ad_p_value": fixed_p,
                    "fixed_ad_accept_5": bool(fixed <= fixed_critical_5),
                    "fixed_ad_accept_15": bool(fixed <= fixed_critical_15),
                    "fixed_ks_statistic": float(ks.statistic),
                    "fixed_ks_p_value": float(ks.pvalue),
                    "fixed_ks_accept_5": bool(float(ks.pvalue) >= 0.05),
                }
            )
        space_rows.append(space_row)
    root = run_root(args.run_id)
    atomic_jsonl(root / "spaces.jsonl", space_rows)
    _atomic_csv(root / "spaces.csv", space_rows)

    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        result = {
            "spaces": len(rows),
            "paper_ad_accept_5": sum(bool(row["paper_ad_accept_5"]) for row in rows),
            "paper_ad_accept_rate_5": float(np.mean([row["paper_ad_accept_5"] for row in rows])),
            "paper_ad_accept_15": sum(bool(row["paper_ad_accept_15"]) for row in rows),
            "paper_ad_accept_rate_15": float(np.mean([row["paper_ad_accept_15"] for row in rows])),
            "mean_of_space_means": float(np.mean([row["sample_mean"] for row in rows])),
            "mean_of_space_stds": float(np.mean([row["sample_std"] for row in rows])),
        }
        if include_fixed_tests:
            result.update(
                {
                    "fixed_ad_accept_5": sum(bool(row["fixed_ad_accept_5"]) for row in rows),
                    "fixed_ad_accept_rate_5": float(
                        np.mean([row["fixed_ad_accept_5"] for row in rows])
                    ),
                    "fixed_ad_accept_15": sum(bool(row["fixed_ad_accept_15"]) for row in rows),
                    "fixed_ad_accept_rate_15": float(
                        np.mean([row["fixed_ad_accept_15"] for row in rows])
                    ),
                    "fixed_ks_accept_5": sum(bool(row["fixed_ks_accept_5"]) for row in rows),
                    "fixed_ks_accept_rate_5": float(
                        np.mean([row["fixed_ks_accept_5"] for row in rows])
                    ),
                }
            )
        return result

    summary = {
        "run_id": args.run_id,
        "primary_definition": "paper-compatible Anderson-Darling normality test at 5%",
        "overall": summarize(space_rows),
        "by_scheme": {
            scheme: summarize([row for row in space_rows if row["watermark"] == scheme])
            for scheme in SCHEMES
            if any(row["watermark"] == scheme for row in space_rows)
        },
    }
    if include_fixed_tests:
        summary.update(
            {
                "fixed_ad_null_critical_5": fixed_critical_5,
                "fixed_ad_null_critical_15": fixed_critical_15,
            }
        )
    atomic_json(root / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def command_manifest(args: argparse.Namespace) -> None:
    root = run_root(args.run_id)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty run directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(args.run_id)
    atomic_json(root / "manifest.json", manifest)
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))


def command_followup_manifest(args: argparse.Namespace) -> None:
    root = run_root(args.run_id)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty run directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    manifest = build_rejected_followup_manifest(
        args.run_id,
        args.source_run_id,
        steps=args.steps,
        selection=args.selection,
        global_seed=args.global_seed,
        verify_source_prefix=args.verify_source_prefix,
    )
    atomic_json(root / "manifest.json", manifest)
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))


def command_useful_sample_manifest(args: argparse.Namespace) -> None:
    root = run_root(args.run_id)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty run directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    manifest = build_useful_parseable_sample_manifest(
        args.run_id,
        source_run_id=args.source_run_id,
    )
    atomic_json(root / "manifest.json", manifest)
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--run-id", default=RUN_ID)
    sub = result.add_subparsers(dest="command", required=True)
    manifest = sub.add_parser("manifest")
    manifest.set_defaults(function=command_manifest)
    followup = sub.add_parser("followup-manifest")
    followup.add_argument("--source-run-id", required=True)
    followup.add_argument("--steps", type=int, default=FOLLOWUP_STEPS)
    followup.add_argument("--global-seed", type=int)
    followup.add_argument(
        "--no-verify-source-prefix",
        action="store_false",
        dest="verify_source_prefix",
    )
    followup.set_defaults(verify_source_prefix=True)
    followup.add_argument(
        "--selection",
        choices=("paper-reject-5", "all"),
        default="paper-reject-5",
    )
    followup.set_defaults(function=command_followup_manifest)
    useful = sub.add_parser("useful-sample-manifest")
    useful.add_argument("--source-run-id", default=USEFUL_SOURCE_RUN_ID)
    useful.set_defaults(function=command_useful_sample_manifest)
    transform = sub.add_parser("transform-array")
    transform.add_argument("--array-index", type=int, required=True)
    transform.add_argument("--array-count", type=int, required=True)
    transform.add_argument("--workers", type=int, default=32)
    transform.set_defaults(function=command_transform_array)
    validate = sub.add_parser("validate")
    validate.set_defaults(function=command_validate)
    score = sub.add_parser("score-group")
    score.add_argument("--scheme", choices=SCHEMES, required=True)
    score.add_argument("--model-slug")
    score.add_argument("--shard-index", type=int, required=True)
    score.add_argument("--shard-count", type=int, required=True)
    score.add_argument("--synthid-device", default="cpu")
    score.set_defaults(function=command_score_group)
    aggregate = sub.add_parser("aggregate")
    aggregate.set_defaults(function=command_aggregate)
    return result


def main() -> None:
    args = parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
