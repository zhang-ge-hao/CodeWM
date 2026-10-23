#!/usr/bin/env python3
"""Recompute added-line precision/recall/F1 from saved patch pairs."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from run_experiment import atomic_write_json, atomic_write_jsonl
from run_python_minifier_postprocess import (
    SCHEMA_VERSION,
    added_line_metrics,
    make_summary,
)


OBSOLETE_FIELDS = (
    "original_changed_diff_lines",
    "final_changed_diff_lines",
    "changed_diff_line_increase_fraction",
    "patch_line_similarity",
)


def update_case(result_path: Path) -> dict[str, Any]:
    case_dir = result_path.parent
    row = json.loads(result_path.read_text())
    original_patch = (case_dir / "original_patch.diff").read_text()
    final_patch = (case_dir / "obfuscated_patch.diff").read_text()
    for field in OBSOLETE_FIELDS:
        row.pop(field, None)
    row.update(added_line_metrics(original_patch, final_patch))
    row["schema_version"] = SCHEMA_VERSION
    atomic_write_json(result_path, row)
    return row


def update_experiment(experiment_dir: Path) -> dict[str, Any]:
    settings = json.loads((experiment_dir / "settings.json").read_text())
    selection = json.loads((experiment_dir / "selection.json").read_text())
    selected_ids = list(selection["instance_ids"])
    result_paths = sorted((experiment_dir / "cases").glob("case-*/result.json"))
    rows = {row["instance_id"]: row for row in map(update_case, result_paths)}
    missing = set(selected_ids) - set(rows)
    if missing:
        raise RuntimeError(f"Missing case results in {experiment_dir}: {sorted(missing)}")
    atomic_write_jsonl(
        experiment_dir / "case_results.jsonl", (rows[value] for value in selected_ids)
    )
    source_dir = Path(settings["source_dir"])
    original_report = json.loads(
        (source_dir / "aggregate_official_report.json").read_text()
    )
    summary = make_summary(
        rows=rows,
        selected_ids=selected_ids,
        original_report=original_report,
        post_report=None,
        settings=settings,
    )
    atomic_write_json(experiment_dir / "summary.json", summary)
    return {
        "experiment": experiment_dir.name,
        "transformed": summary["transformed_cases"],
        **summary["added_line_similarity"]["successfully_transformed_cases"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")
    return args


def main() -> None:
    args = parse_args()
    experiments = sorted(
        path
        for path in args.root.iterdir()
        if path.is_dir() and (path / "settings.json").is_file()
    )
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=min(args.workers, len(experiments))) as pool:
        futures = {pool.submit(update_experiment, path): path for path in experiments}
        for future in as_completed(futures):
            results.append(future.result())
    print(json.dumps(sorted(results, key=lambda row: row["experiment"]), indent=2))


if __name__ == "__main__":
    main()
