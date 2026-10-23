#!/usr/bin/env python3
"""Dependency-light transform entry point for the system libcst environment."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

from .common import load_manifest
from .transform import refresh_baseline_shard, transform_shard


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="rw100-useful-v1")
    parser.add_argument("--index", type=int)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--memory-mb", type=int, default=1024)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="number of consecutive manifest shards processed by this array task",
    )
    args = parser.parse_args(argv)
    index = args.index
    if index is None:
        value = os.environ.get("SLURM_ARRAY_TASK_ID")
        if value is None:
            raise ValueError("--index or SLURM_ARRAY_TASK_ID is required")
        index = int(value)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    operation = refresh_baseline_shard if args.baseline_only else transform_shard
    kwargs = {
        "timeout": args.timeout,
        "memory_mb": args.memory_mb,
    }
    if not args.baseline_only:
        kwargs["overwrite"] = args.overwrite
    shard_count = len(load_manifest(args.run_id)["shards"])
    start = index * args.batch_size
    stop = min(start + args.batch_size, shard_count)
    if start >= shard_count:
        raise IndexError(
            f"batch {index} starts at shard {start}, outside [0, {shard_count})"
        )
    indices = tuple(range(start, stop))
    if len(indices) == 1:
        print(operation(args.run_id, indices[0], **kwargs))
        return 0

    failures: list[tuple[int, BaseException]] = []
    with ProcessPoolExecutor(max_workers=len(indices)) as executor:
        futures = {
            executor.submit(operation, args.run_id, shard, **kwargs): shard
            for shard in indices
        }
        for future in as_completed(futures):
            shard = futures[future]
            try:
                print(f"shard={shard} output={future.result()}", flush=True)
            except BaseException as error:
                failures.append((shard, error))
                print(
                    f"shard={shard} failed={type(error).__name__}: {error}",
                    flush=True,
                )
    if failures:
        detail = "; ".join(
            f"{shard}:{type(error).__name__}:{error}"
            for shard, error in sorted(failures)
        )
        raise RuntimeError(f"{len(failures)} transform shards failed: {detail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
