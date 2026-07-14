"""Build a compact, machine-readable status report for score shard outputs."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

from run import NEGATIVE_VARIANTS, POSITIVE_VARIANTS, build_score_shards
from source_data import load_dataset_inputs, read_jsonl, write_jsonl


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "data" / "shard_summary.jsonl"


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def summarize() -> list[dict[str, Any]]:
    inputs_by_dataset = {
        dataset: load_dataset_inputs(dataset)
        for dataset in {shard.dataset for shard in build_score_shards()}
    }
    summaries: list[dict[str, Any]] = []
    for shard in build_score_shards():
        expected_tasks = list(
            inputs_by_dataset[shard.dataset].task_names[
                shard.task_start : shard.task_stop
            ]
        )
        errors: list[str] = []
        rows: list[dict[str, Any]] = []
        if not shard.output_path.is_file():
            errors.append("missing_output")
        else:
            try:
                rows = read_jsonl(shard.output_path)
            except (OSError, ValueError) as exc:
                errors.append(f"unreadable_output:{exc}")

        actual_tasks = [row.get("task") for row in rows]
        if rows and actual_tasks != expected_tasks:
            errors.append("task_order_or_coverage_mismatch")
        if rows and any(
            row.get("dataset") != shard.dataset
            or row.get("config") != shard.config_id
            for row in rows
        ):
            errors.append("dataset_or_config_mismatch")
        if rows and any(
            row.get("score_shard", {}).get("job_index") != shard.global_index
            for row in rows
        ):
            errors.append("job_index_mismatch")

        positive = {
            name: {
                "finite_z": sum(
                    _finite(row.get("positive", {}).get(name, {}).get("z_score"))
                    for row in rows
                ),
                "missing_z": sum(
                    not _finite(
                        row.get("positive", {}).get(name, {}).get("z_score")
                    )
                    for row in rows
                ),
            }
            for name in POSITIVE_VARIANTS
        }
        negative = {
            name: {
                "finite_z": sum(
                    _finite(row.get("negative", {}).get(name, {}).get("z_score"))
                    for row in rows
                ),
                "invalid": sum(
                    row.get("negative", {}).get(name, {}).get("invalid") is True
                    for row in rows
                ),
                "missing_z": sum(
                    not _finite(
                        row.get("negative", {}).get(name, {}).get("z_score")
                    )
                    for row in rows
                ),
            }
            for name in NEGATIVE_VARIANTS
        }
        summaries.append(
            {
                "job_index": shard.global_index,
                "dataset": shard.dataset,
                "config": shard.config_id,
                "part": shard.part_index,
                "task_start": shard.task_start,
                "task_stop": shard.task_stop,
                "expected_rows": shard.task_count,
                "actual_rows": len(rows),
                "output": str(shard.output_path),
                "valid": not errors and len(rows) == shard.task_count,
                "errors": errors,
                "positive": positive,
                "negative": negative,
            }
        )
    return summaries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    rows = summarize()
    write_jsonl(args.output, rows, overwrite=args.overwrite)
    valid = sum(row["valid"] for row in rows)
    missing = sum("missing_output" in row["errors"] for row in rows)
    invalid = len(rows) - valid - missing
    print(
        f"wrote {args.output}: total={len(rows)} valid={valid} "
        f"missing={missing} invalid={invalid}"
    )
    return 0 if invalid == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
