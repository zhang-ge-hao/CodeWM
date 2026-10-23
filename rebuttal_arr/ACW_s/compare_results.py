#!/usr/bin/env python3
"""Compare official-runner metrics with the ACW-s values in paper Table 1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PAPER_TARGETS = {
    "humaneval": {
        "avg_pass_1": 0.6413,
        "avg_pass_10": 0.7639,
        "roc_auc": 0.9338,
        "tpr_at_fpr_05": 0.6187,
    },
    "mbpp": {
        "avg_pass_1": 0.4064,
        "avg_pass_10": 0.4833,
        "roc_auc": 0.8891,
        "tpr_at_fpr_05": 0.5480,
    },
}


def newest_metrics(task_dir: Path) -> Path:
    candidates = list(task_dir.glob("*/results/metrics.json"))
    # The pinned upstream parser lower-cases all string arguments. Support the
    # path produced by an interrupted/legacy run as well as the corrected path.
    lowercase_task_dir = Path(str(task_dir).replace("/ACW_s/", "/acw_s/"))
    if lowercase_task_dir != task_dir:
        candidates.extend(lowercase_task_dir.glob("*/results/metrics.json"))
    if not candidates:
        raise FileNotFoundError(f"no metrics.json found below {task_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def compare_task(task: str, run_root: Path) -> dict[str, Any]:
    skip_marker = run_root / f"SKIP_{task.upper()}"
    if skip_marker.exists():
        return {"skipped": True, "skip_marker": str(skip_marker)}
    metrics_path = newest_metrics(run_root / task)
    actual = json.loads(metrics_path.read_text())
    target = PAPER_TARGETS[task]
    comparison = {}
    for metric, expected in target.items():
        observed = float(actual[metric])
        comparison[metric] = {
            "observed": observed,
            "paper": expected,
            "absolute_difference": observed - expected,
            "percentage_point_difference": 100.0 * (observed - expected),
        }
    return {
        "metrics_path": str(metrics_path),
        "reported_sample_count": actual.get("total_count"),
        "comparison": comparison,
        "all_metrics": actual,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--tasks", default="humaneval,mbpp", help="comma-separated task names"
    )
    args = parser.parse_args()

    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    report = {
        "paper_table": "Table 1, OpenCoder-1.5B-Instruct, ACW-s",
        "tasks": {task: compare_task(task, args.run_root) for task in tasks},
    }
    output = args.run_root / "comparison.json"
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
