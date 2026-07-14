#!/usr/bin/env python3
"""Resumable all-case equivalent-space distribution experiment."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import secrets
import sys
from typing import Any, Iterable, Mapping

from experiment.common import (
    atomic_json,
    atomic_jsonl,
    iter_jsonl,
    relative_to_repo,
    resolve_repo_path,
    sha256_file,
)

from .run import (
    DATA_ROOT,
    GLOBAL_SEED,
    NEW_PROTOTYPE_ROOT,
    SCHEMES,
    TRAJECTORIES_PER_SEED,
    _anderson_critical,
    discover_candidates,
    iter_transform_rows,
    load_manifest,
    run_root,
)


RUN_ID = "rw100-z4-all1620-random-seeds-v1"
STEPS = 100
TARGET_NAME = "results.jsonl"
TRANSFORM_TASKS_PER_SHARD = 10


def target_path(run_id: str) -> Path:
    return run_root(run_id) / TARGET_NAME


def _existing_results(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    result: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        key = row.get("key")
        if not isinstance(key, str) or not key:
            raise ValueError(f"result without a valid key in {path}")
        if key in result:
            raise ValueError(f"duplicate result key {key!r} in {path}")
        if "timestamp" in row or "created_utc" in row or "updated_utc" in row:
            raise ValueError(f"timestamp field is forbidden in {path}: {key}")
        if not isinstance(row.get("random_seed"), int):
            raise ValueError(f"result without integer random_seed in {path}: {key}")
        result[key] = row
    return result


def write_sorted_results(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    ordered = sorted((dict(row) for row in rows), key=lambda row: str(row["key"]))
    keys = [str(row["key"]) for row in ordered]
    if len(keys) != len(set(keys)):
        raise ValueError("cannot write duplicate result keys")
    atomic_jsonl(path, ordered, overwrite=path.exists())


def build_manifest(run_id: str) -> dict[str, Any]:
    candidates, all_configs = discover_candidates()
    existing = _existing_results(target_path(run_id))
    candidate_keys = {str(candidate["candidate_key"]) for candidate in candidates}
    unknown = set(existing) - candidate_keys
    if unknown:
        raise ValueError(f"target contains {len(unknown)} unknown keys")

    used_seeds = {int(row["random_seed"]) for row in existing.values()}
    if len(used_seeds) != len(existing):
        raise ValueError("existing target contains duplicate random seeds")
    for candidate in candidates:
        key = str(candidate["candidate_key"])
        if key in existing:
            seed = int(existing[key]["random_seed"])
        else:
            seed = secrets.randbits(63)
            while seed == 0 or seed in used_seeds:
                seed = secrets.randbits(63)
            used_seeds.add(seed)
        candidate["random_seed"] = seed

    candidates.sort(key=lambda row: str(row["candidate_key"]))
    config_keys = sorted({str(row["config_key"]) for row in candidates})
    configs = [all_configs[key] for key in config_keys]
    trajectory_count = len(candidates) * TRAJECTORIES_PER_SEED
    shards = [
        {
            "index": index,
            "task_start": start,
            "task_stop": min(start + TRANSFORM_TASKS_PER_SHARD, trajectory_count),
        }
        for index, start in enumerate(
            range(0, trajectory_count, TRANSFORM_TASKS_PER_SHARD)
        )
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
    cases_by_scheme = Counter(str(row["watermark"]) for row in candidates)
    configs_by_scheme = Counter(str(row["watermark"]) for row in configs)
    cases_by_model = Counter(str(row["model_slug"]) for row in candidates)
    cases_by_dataset = Counter(str(row["dataset"]) for row in candidates)
    return {
        "schema_version": 1,
        "run_id": run_id,
        "selection": {
            "universe": "all real Python watermark configurations",
            "saved_original_passed": True,
            "original_z_score_strictly_above": 4.0,
            "candidate_count": len(candidates),
            "sample_without_replacement": False,
        },
        "walk": {
            "rule_profile": "full",
            "steps": STEPS,
            "trajectories_per_seed": TRAJECTORIES_PER_SEED,
            "global_seed": 0,
            "seed_policy": "independent system-random case seed, then deterministic per-trajectory derivation",
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
        "output": {
            "path": relative_to_repo(target_path(run_id)),
            "unique_key": "key",
            "sort": "lexicographic ascending by key before every write",
            "timestamps": False,
            "resume_from_existing_target": True,
        },
        "candidates": candidates,
        "configs": configs,
        "shards": shards,
        "counts": {
            "cases": len(candidates),
            "cases_by_scheme": dict(sorted(cases_by_scheme.items())),
            "configs": len(configs),
            "configs_by_scheme": dict(sorted(configs_by_scheme.items())),
            "cases_by_model": dict(sorted(cases_by_model.items())),
            "cases_by_dataset": dict(sorted(cases_by_dataset.items())),
            "trajectories": trajectory_count,
            "transitions": trajectory_count * STEPS,
            "saved_programs": trajectory_count * (STEPS + 1),
            "transform_shards": len(shards),
            "existing_target_results_at_manifest_creation": len(existing),
            "missing_target_results_at_manifest_creation": len(candidates) - len(existing),
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


def command_manifest(args: argparse.Namespace) -> None:
    root = run_root(args.run_id)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "manifest.json"
    if path.is_file():
        manifest = load_manifest(args.run_id)
        test_contract = manifest["test"]
        if test_contract.get("require_baseline_passed") is not False or (
            test_contract.get("require_same_test_outcome") is not True
        ):
            test_contract["require_baseline_passed"] = False
            test_contract["require_same_test_outcome"] = True
            atomic_json(path, manifest, overwrite=True)
        existing = _existing_results(target_path(args.run_id))
        candidate_map = {
            str(candidate["candidate_key"]): candidate
            for candidate in manifest["candidates"]
        }
        if set(existing) - set(candidate_map):
            raise ValueError("target contains keys outside the frozen manifest")
        for key, row in existing.items():
            if int(row["random_seed"]) != int(candidate_map[key]["random_seed"]):
                raise ValueError(f"target/manifest random seed mismatch for {key}")
        print(
            json.dumps(
                {
                    "manifest": "existing",
                    "target_results": len(existing),
                    "missing_results": len(candidate_map) - len(existing),
                    "counts": manifest["counts"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    manifest = build_manifest(args.run_id)
    atomic_json(path, manifest)
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))


def _result_row(
    candidate: Mapping[str, Any],
    score_rows: list[dict[str, Any]],
    endpoint_programs: Mapping[int, str],
) -> dict[str, Any]:
    import numpy as np
    from scipy import stats

    ordered = sorted(score_rows, key=lambda row: int(row["trajectory_index"]))
    if len(ordered) != TRAJECTORIES_PER_SEED:
        raise ValueError(f"{candidate['candidate_key']}: {len(ordered)} scores")
    indices = [int(row["trajectory_index"]) for row in ordered]
    if indices != list(range(TRAJECTORIES_PER_SEED)):
        raise ValueError(f"{candidate['candidate_key']}: trajectory index mismatch")
    values = [float(row["score"]["z_score"]) for row in ordered]
    if sorted(endpoint_programs) != indices:
        raise ValueError(
            f"{candidate['candidate_key']}: endpoint program index mismatch"
        )
    programs = [endpoint_programs[index] for index in indices]
    baseline_outcomes = {
        row.get("baseline_execution_passed") for row in ordered
    }
    final_outcomes = {row.get("final_execution_passed") for row in ordered}
    saved_outcomes = {row.get("saved_original_passed") for row in ordered}
    if len(baseline_outcomes) != 1 or len(final_outcomes) != 1 or len(saved_outcomes) != 1:
        raise ValueError(f"{candidate['candidate_key']}: inconsistent execution metadata")
    baseline_passed = next(iter(baseline_outcomes))
    final_passed = next(iter(final_outcomes))
    saved_passed = next(iter(saved_outcomes))
    ad = stats.anderson(values, dist="norm")
    critical_5 = _anderson_critical(ad, 5.0)
    critical_15 = _anderson_critical(ad, 15.0)
    return {
        "key": candidate["candidate_key"],
        "random_seed": int(candidate["random_seed"]),
        "record_id": candidate["record_id"],
        "config_key": candidate["config_key"],
        "task_name": candidate["task_name"],
        "watermark": candidate["watermark"],
        "model_slug": candidate["model_slug"],
        "dataset": candidate["dataset"],
        "saved_original_passed": saved_passed,
        "baseline_execution_passed": baseline_passed,
        "final_execution_passed": final_passed,
        "test_outcome_preserved": baseline_passed == final_passed,
        "original_z_score": float(candidate["original_z_score"]),
        "steps": STEPS,
        "trajectories": TRAJECTORIES_PER_SEED,
        "endpoint_z_scores": values,
        "endpoint_programs": programs,
        "sample_mean": float(np.mean(values)),
        "sample_std": float(np.std(values, ddof=1)),
        "sample_min": min(values),
        "sample_max": max(values),
        "paper_ad_statistic": float(ad.statistic),
        "paper_ad_critical_5": critical_5,
        "paper_ad_accept_5": bool(float(ad.statistic) <= critical_5),
        "paper_ad_critical_15": critical_15,
        "paper_ad_accept_15": bool(float(ad.statistic) <= critical_15),
    }


def _load_endpoint_programs(run_id: str) -> dict[str, dict[int, str]]:
    """Load only each trajectory's step-100 program from saved walk records."""
    result: dict[str, dict[int, str]] = defaultdict(dict)
    for row in iter_transform_rows(run_id):
        if row.get("status") != "ok":
            continue
        candidate_key = str(row["candidate_key"])
        trajectory_index = int(row["trajectory_index"])
        programs = row.get("programs")
        if not isinstance(programs, list) or len(programs) != STEPS + 1:
            raise ValueError(
                f"{candidate_key} trajectory {trajectory_index}: expected "
                f"{STEPS + 1} saved programs"
            )
        endpoint = programs[-1]
        if not isinstance(endpoint, str):
            raise ValueError(
                f"{candidate_key} trajectory {trajectory_index}: invalid endpoint program"
            )
        candidate_programs = result[candidate_key]
        if trajectory_index in candidate_programs:
            raise ValueError(
                f"{candidate_key} trajectory {trajectory_index}: duplicate endpoint program"
            )
        candidate_programs[trajectory_index] = endpoint
    return dict(result)


def command_aggregate(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.run_id)
    candidates = {
        str(candidate["candidate_key"]): candidate
        for candidate in manifest["candidates"]
    }
    target = target_path(args.run_id)
    merged = _existing_results(target)
    if set(merged) - set(candidates):
        raise ValueError("target contains unknown result keys")
    for key, row in merged.items():
        if int(row["random_seed"]) != int(candidates[key]["random_seed"]):
            raise ValueError(f"target/manifest random seed mismatch for {key}")

    by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted((run_root(args.run_id) / "scores").glob("*.jsonl.gz")):
        for row in iter_jsonl(path):
            if row.get("status") == "ok" and row.get("score"):
                by_candidate[str(row["candidate_key"])].append(row)
    fallback_endpoint_programs: dict[str, dict[int, str]] | None = None
    for key in sorted(by_candidate):
        if key not in candidates:
            raise ValueError(f"score contains unknown candidate {key}")
        if len(by_candidate[key]) == TRAJECTORIES_PER_SEED:
            programs_from_scores = {
                int(row["trajectory_index"]): str(row["endpoint_program"])
                for row in by_candidate[key]
                if isinstance(row.get("endpoint_program"), str)
            }
            if len(programs_from_scores) != TRAJECTORIES_PER_SEED:
                if fallback_endpoint_programs is None:
                    fallback_endpoint_programs = _load_endpoint_programs(args.run_id)
                programs_from_scores = fallback_endpoint_programs.get(key, {})
            merged[key] = _result_row(
                candidates[key],
                by_candidate[key],
                programs_from_scores,
            )

    write_sorted_results(target, merged.values())
    missing = sorted(set(candidates) - set(merged))
    summary = {
        "target": str(target),
        "written": len(merged),
        "missing": len(missing),
        "paper_ad_accept_5": sum(
            bool(row["paper_ad_accept_5"]) for row in merged.values()
        ),
        "paper_ad_accept_15": sum(
            bool(row["paper_ad_accept_15"]) for row in merged.values()
        ),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.require_complete and missing:
        raise RuntimeError(
            f"wrote resumable partial target with {len(missing)} missing cases"
        )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--run-id", default=RUN_ID)
    sub = result.add_subparsers(dest="command", required=True)
    manifest = sub.add_parser("manifest")
    manifest.set_defaults(function=command_manifest)
    aggregate = sub.add_parser("aggregate")
    aggregate.add_argument("--require-complete", action="store_true")
    aggregate.set_defaults(function=command_aggregate)
    return result


def main() -> None:
    args = parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
