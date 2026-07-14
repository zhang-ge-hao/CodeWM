"""Re-run the previous experiment's twenty highest-z cases with the new walk."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics
from typing import Any, Mapping

from .common import (
    atomic_json,
    atomic_jsonl,
    config_map,
    index_by_task,
    iter_jsonl,
    load_manifest,
    load_transforms_for_config,
    resolve_repo_path,
    run_root,
)
SOURCE_RUN_ID = "rw100-useful-v1"
FOLLOWUP_RUN_ID = "high-z-top20-rw100-v2"
TOP_K = 20
STEPS = 100
GLOBAL_SEED = 10772


def select_high_z_cases(
    source_run_id: str = SOURCE_RUN_ID,
    *,
    top_k: int = TOP_K,
) -> list[dict[str, Any]]:
    source_root = run_root(source_run_id)
    candidates: list[dict[str, Any]] = []
    for path in sorted((source_root / "scores").glob("*.jsonl.gz")):
        for row in iter_jsonl(path):
            score = row.get("score")
            if row.get("status") != "ok" or not isinstance(score, Mapping):
                continue
            z_score = score.get("z_score")
            if z_score is None:
                continue
            candidates.append(
                {
                    "config_key": str(row["config_key"]),
                    "watermark": str(row["watermark"]),
                    "dataset": str(row["dataset"]),
                    "task_name": str(row["task_name"]),
                    "record_id": str(row["record_id"]),
                    "old_step100_z": float(z_score),
                    "old_score_file": path.name,
                }
            )
    candidates.sort(
        key=lambda row: (
            -float(row["old_step100_z"]),
            str(row["config_key"]),
            str(row["task_name"]),
        )
    )
    selected = candidates[:top_k]
    if len(selected) != top_k:
        raise ValueError(f"expected {top_k} high-z cases, found {len(selected)}")
    for rank, row in enumerate(selected, start=1):
        row["rank"] = rank
    return selected


def _generation_records(
    configs: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for key, config in configs.items():
        path = resolve_repo_path(str(config["generate_path"]))
        result[key] = index_by_task(iter_jsonl(path), str(path))
    return result


def transform_selected(
    *,
    source_run_id: str = SOURCE_RUN_ID,
    followup_run_id: str = FOLLOWUP_RUN_ID,
    overwrite: bool = False,
) -> Path:
    from .transform import transform_record

    manifest = load_manifest(source_run_id)
    configs = config_map(manifest)
    selected = select_high_z_cases(source_run_id)
    selected_configs = {
        str(row["config_key"]): configs[str(row["config_key"])]
        for row in selected
    }
    generations = _generation_records(selected_configs)
    output_root = run_root(followup_run_id)
    selection_path = output_root / "selection.json"
    transform_path = output_root / "transforms.jsonl.gz"
    atomic_json(
        selection_path,
        {
            "schema_version": 1,
            "source_run_id": source_run_id,
            "followup_run_id": followup_run_id,
            "selection": "global top-20 successful old step-100 z-scores",
            "top_k": TOP_K,
            "steps": STEPS,
            "global_seed": GLOBAL_SEED,
            "scheme_counts": dict(Counter(row["watermark"] for row in selected)),
            "cases": selected,
        },
        overwrite=overwrite,
    )

    def rows():
        for item in selected:
            key = str(item["config_key"])
            task = str(item["task_name"])
            record = generations[key][task]
            if str(record["id"]) != item["record_id"]:
                raise ValueError(f"record id mismatch for {key}/{task}")
            transformed = transform_record(
                record,
                config=configs[key],
                steps=STEPS,
                global_seed=GLOBAL_SEED,
                timeout=10.0,
                memory_mb=1024,
            )
            yield {**item, **transformed}

    atomic_jsonl(transform_path, rows(), overwrite=overwrite)
    return transform_path


def _write_report(path: Path, rows: list[dict[str, Any]], summary: Mapping[str, Any]) -> None:
    lines = [
        "# High-z top-20 follow-up",
        "",
        f"- Source run: `{SOURCE_RUN_ID}`",
        f"- New obfuscator walk: `{STEPS}` steps, one trajectory per case",
        f"- Mean old step-100 z: {summary['old_z_mean']:.6f}",
        f"- Mean new step-100 z: {summary['new_z_mean']:.6f}",
        f"- Median old/new z: {summary['old_z_median']:.6f} / {summary['new_z_median']:.6f}",
        f"- z > 4: {summary['old_z_gt_4']}/{TOP_K} -> {summary['new_z_gt_4']}/{TOP_K}",
        "",
        "| Rank | Scheme | Config | Task | Old z | New z | Delta | Test |",
        "|---:|---|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        baseline_passed = row.get("baseline_passed")
        final_passed = row.get("final_passed")
        if baseline_passed is True:
            test_label = "preserved" if final_passed is True else "REGRESSION"
        else:
            test_label = "improved" if final_passed is True else "baseline-failed"
        lines.append(
            f"| {row['rank']} | {row['watermark']} | `{row['config_key']}` | "
            f"`{row['task_name']}` | {row['old_step100_z']:.6f} | "
            f"{row['new_step100_z']:.6f} | {row['z_delta']:.6f} | {test_label} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_top_code_comparison(
    output_root: Path,
    rows: list[dict[str, Any]],
    *,
    count: int = 3,
) -> Path:
    transforms = {
        str(row["record_id"]): row
        for row in iter_jsonl(output_root / "transforms.jsonl.gz")
    }
    selected = sorted(
        rows,
        key=lambda row: float(row["new_step100_z"]),
        reverse=True,
    )[:count]
    lines = ["# Top-3 new-z code comparison", ""]
    for position, row in enumerate(selected, start=1):
        transformed = transforms[str(row["record_id"])]
        before = str(transformed["programs"][0])
        after = str(transformed["programs"][-1])
        lines.extend(
            [
                f"## {position}. {row['watermark'].upper()} — {row['task_name']}",
                "",
                f"- Config: `{row['config_key']}`",
                f"- Old step-100 z: `{float(row['old_step100_z']):.12f}`",
                f"- New step-100 z: `{float(row['new_step100_z']):.12f}`",
                f"- Bytes: `{len(before.encode('utf-8'))} -> {len(after.encode('utf-8'))}`",
                f"- Rule counts: `{json.dumps(transformed['rule_counts'], sort_keys=True)}`",
                "",
                "### Step 0",
                "",
                "```python",
                before.rstrip(),
                "```",
                "",
                "### Step 100",
                "",
                "```python",
                after.rstrip(),
                "```",
                "",
            ]
        )
    path = output_root / "top3_code_comparison.md"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(path)
    return path


def refresh_report(*, followup_run_id: str = FOLLOWUP_RUN_ID) -> Path:
    output_root = run_root(followup_run_id)
    rows = list(iter_jsonl(output_root / "results.jsonl"))
    summary_path = output_root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["baseline_passed"] = sum(row["baseline_passed"] is True for row in rows)
    summary["final_passed"] = sum(row["final_passed"] is True for row in rows)
    summary["test_outcomes"] = dict(
        Counter(
            "preserved"
            if row["baseline_passed"] is True and row["final_passed"] is True
            else "regression"
            if row["baseline_passed"] is True
            else "improved"
            if row["final_passed"] is True
            else "baseline_failed"
            for row in rows
        )
    )
    by_scheme = {}
    for scheme in sorted({str(row["watermark"]) for row in rows}):
        selected = [row for row in rows if row["watermark"] == scheme]
        old_values = [float(row["old_step100_z"]) for row in selected]
        new_values = [float(row["new_step100_z"]) for row in selected]
        by_scheme[scheme] = {
            "cases": len(selected),
            "old_z_mean": statistics.mean(old_values),
            "new_z_mean": statistics.mean(new_values),
            "old_z_gt_4": sum(value > 4.0 for value in old_values),
            "new_z_gt_4": sum(value > 4.0 for value in new_values),
        }
    summary["by_scheme"] = by_scheme
    atomic_json(summary_path, summary, overwrite=True)
    report_path = output_root / "report.md"
    _write_report(report_path, rows, summary)
    _write_top_code_comparison(output_root, rows)
    return report_path


def score_selected(
    *,
    source_run_id: str = SOURCE_RUN_ID,
    followup_run_id: str = FOLLOWUP_RUN_ID,
    overwrite: bool = False,
) -> Path:
    from .detectors import make_scorer
    from .run import _check_saved_score

    manifest = load_manifest(source_run_id)
    configs = config_map(manifest)
    transformed_rows = list(iter_jsonl(run_root(followup_run_id) / "transforms.jsonl.gz"))
    if len(transformed_rows) != TOP_K:
        raise ValueError(f"expected {TOP_K} transformed rows, found {len(transformed_rows)}")
    selected_configs = {
        str(row["config_key"]): configs[str(row["config_key"])]
        for row in transformed_rows
    }
    generations = _generation_records(selected_configs)
    old_transforms = {
        key: load_transforms_for_config(source_run_id, key)
        for key in selected_configs
    }
    scorers: dict[str, Any] = {}
    validated: set[str] = set()
    results: list[dict[str, Any]] = []
    for transformed in sorted(transformed_rows, key=lambda row: int(row["rank"])):
        if transformed.get("status") != "ok":
            raise RuntimeError(
                f"transform failed for rank {transformed['rank']}: "
                f"{transformed.get('error_type')}: {transformed.get('error')}"
            )
        key = str(transformed["config_key"])
        task = str(transformed["task_name"])
        config = selected_configs[key]
        record = generations[key][task]
        scorer = scorers.get(key)
        if scorer is None:
            scorer = make_scorer(config, synthid_device="cuda")
            scorers[key] = scorer
        if key not in validated:
            first_record = next(iter(generations[key].values()))
            _check_saved_score(scorer, config, first_record)
            validated.add(key)
        old_transform = old_transforms[key][task]
        old_recomputed = scorer.score(record, str(old_transform["detection_g4d"]))
        old_saved = float(transformed["old_step100_z"])
        old_drift = float(old_recomputed["z_score"]) - old_saved
        if not math.isclose(old_drift, 0.0, rel_tol=0.0, abs_tol=1e-10):
            raise RuntimeError(
                f"old step-100 score drift for {key}/{task}: "
                f"saved={old_saved}, recomputed={old_recomputed['z_score']}"
            )
        new_score = scorer.score(record, str(transformed["detection_g4d"]))
        new_z = float(new_score["z_score"])
        results.append(
            {
                "rank": int(transformed["rank"]),
                "watermark": str(transformed["watermark"]),
                "config_key": key,
                "task_name": task,
                "record_id": str(transformed["record_id"]),
                "seed": int(transformed["seed"]),
                "baseline_z_score": float(record["z_score"]),
                "old_step100_z": old_saved,
                "old_step100_recomputed_z": float(old_recomputed["z_score"]),
                "old_score_abs_drift": abs(old_drift),
                "new_step100_z": new_z,
                "z_delta": new_z - old_saved,
                "baseline_passed": transformed["baseline"]["execution"]["passed"],
                "final_passed": transformed["final"]["execution"]["passed"],
                "source_bytes": len(transformed["programs"][0].encode("utf-8")),
                "final_bytes": int(transformed["final"]["bytes"]),
                "rule_counts": transformed["rule_counts"],
                "score": new_score,
            }
        )

    old_z = [float(row["old_step100_z"]) for row in results]
    new_z = [float(row["new_step100_z"]) for row in results]
    summary = {
        "schema_version": 1,
        "source_run_id": source_run_id,
        "followup_run_id": followup_run_id,
        "cases": len(results),
        "scheme_counts": dict(Counter(row["watermark"] for row in results)),
        "old_z_mean": statistics.mean(old_z),
        "new_z_mean": statistics.mean(new_z),
        "old_z_median": statistics.median(old_z),
        "new_z_median": statistics.median(new_z),
        "old_z_min": min(old_z),
        "new_z_min": min(new_z),
        "old_z_max": max(old_z),
        "new_z_max": max(new_z),
        "old_z_gt_4": sum(value > 4.0 for value in old_z),
        "new_z_gt_4": sum(value > 4.0 for value in new_z),
        "semantic_regressions": sum(
            row["baseline_passed"] is True and row["final_passed"] is False
            for row in results
        ),
        "transform_errors": 0,
        "max_old_score_abs_drift": max(row["old_score_abs_drift"] for row in results),
    }
    output_root = run_root(followup_run_id)
    result_path = output_root / "results.jsonl"
    atomic_jsonl(result_path, results, overwrite=overwrite)
    atomic_json(output_root / "summary.json", summary, overwrite=overwrite)
    _write_report(output_root / "report.md", results, summary)
    _write_top_code_comparison(output_root, results)
    return result_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("transform", "score", "report"))
    parser.add_argument("--source-run-id", default=SOURCE_RUN_ID)
    parser.add_argument("--followup-run-id", default=FOLLOWUP_RUN_ID)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "transform":
        output = transform_selected(
            source_run_id=args.source_run_id,
            followup_run_id=args.followup_run_id,
            overwrite=args.overwrite,
        )
    elif args.command == "score":
        output = score_selected(
            source_run_id=args.source_run_id,
            followup_run_id=args.followup_run_id,
            overwrite=args.overwrite,
        )
    else:
        output = refresh_report(followup_run_id=args.followup_run_id)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
