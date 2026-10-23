#!/usr/bin/env python3
"""Prepare and execute the ACW-s HumanEval Python-Minifier CPU stage.

The completed upstream run persisted 20 generations for 160 tasks but silently
dropped four prompt-only tasks.  This runner reconstructs those four exact
postprocessed outputs as 20 copies of the benchmark prompt, keeps them in the
164-task denominator, applies Python-Minifier, and executes the HumanEval tests.
"""

from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
HUMANEVAL_PATH = REPO_ROOT / "data" / "original" / "humaneval-x_py.jsonl"
EXPECTED_TASKS = 164
EXPECTED_COMPLETIONS = 20

PYTHON_IMPORT_HELPER = """\
import math
import re
import sys
import copy
import datetime
import itertools
import collections
import heapq
import functools
import hashlib
import string
from typing import *
from collections import *
try:
    import numpy
    import numpy as np
except ImportError:
    pass
"""


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def entry_point(prompt: str) -> str:
    tree = ast.parse(prompt)
    functions = [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if not functions:
        raise ValueError("HumanEval prompt has no top-level function")
    return functions[-1]


def pass_flags(sample: dict[str, Any]) -> list[bool]:
    pass_results = sample["metrics"]["pass_results"]
    if not isinstance(pass_results, dict) or len(pass_results) != 1:
        raise ValueError("unexpected upstream pass_results structure")
    pairs = next(iter(pass_results.values()))
    flags: dict[int, bool] = {}
    for pair in pairs:
        completion_id, result = pair
        flags[int(completion_id)] = bool(result["passed"])
    expected = set(range(EXPECTED_COMPLETIONS))
    if set(flags) != expected:
        raise ValueError(f"upstream completion IDs differ from {sorted(expected)}")
    return [flags[index] for index in range(EXPECTED_COMPLETIONS)]


def load_task_inputs(
    source_results: Path,
) -> list[dict[str, Any]]:
    benchmark = read_jsonl(HUMANEVAL_PATH)
    if len(benchmark) != EXPECTED_TASKS:
        raise ValueError(
            f"expected {EXPECTED_TASKS} HumanEval rows, found {len(benchmark)}"
        )

    tasks: list[dict[str, Any]] = []
    for index, benchmark_row in enumerate(benchmark):
        sample_path = source_results / f"sample_{index}.json"
        reconstructed_empty = not sample_path.is_file()
        if reconstructed_empty:
            prompt = str(benchmark_row["prompt"]).strip()
            generations = [prompt] * EXPECTED_COMPLETIONS
            baseline_passed = [False] * EXPECTED_COMPLETIONS
        else:
            sample = read_json(sample_path)
            prompt_value = sample["prompt"]
            prompt = (
                str(prompt_value[0])
                if isinstance(prompt_value, list)
                else str(prompt_value)
            )
            generations = sample["generations"]
            if len(generations) != EXPECTED_COMPLETIONS:
                raise ValueError(
                    f"{sample_path} has {len(generations)} completions, "
                    f"expected {EXPECTED_COMPLETIONS}"
                )
            baseline_passed = pass_flags(sample)

        expected_prompt = str(benchmark_row["prompt"]).strip()
        if prompt != expected_prompt:
            raise ValueError(f"prompt mismatch for HumanEval task {index}")
        tasks.append(
            {
                "task_index": index,
                "source_task_id": benchmark_row["task_id"],
                "prompt": prompt,
                "test": benchmark_row["test"],
                "entry_point": entry_point(prompt),
                "generations": generations,
                "baseline_passed": baseline_passed,
                "reconstructed_empty": reconstructed_empty,
            }
        )
    return tasks


def limit_resources(timeout: float, memory_mb: int) -> None:
    cpu_seconds = max(1, int(math.ceil(timeout)) + 1)
    memory_bytes = memory_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    resource.setrlimit(resource.RLIMIT_FSIZE, (8 * 1024 * 1024, 8 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))


def execute_humaneval(
    source: str,
    test: str,
    function_name: str,
    *,
    timeout: float,
    memory_mb: int,
    work_root: Path,
) -> dict[str, Any]:
    script = (
        PYTHON_IMPORT_HELPER.rstrip()
        + "\n\n"
        + source.rstrip()
        + "\n\n"
        + test.lstrip()
        + f"\ncheck({function_name})\n"
    )
    started = time.monotonic()
    work_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="case-", dir=work_root) as directory:
        try:
            completed = subprocess.run(
                [sys.executable, "-s", "-c", script],
                cwd=directory,
                env={
                    "PATH": os.environ.get("PATH", ""),
                    "PYTHONHASHSEED": "0",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                },
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                preexec_fn=lambda: limit_resources(timeout, memory_mb),
            )
            return {
                "passed": completed.returncode == 0,
                "returncode": completed.returncode,
                "timed_out": False,
                "duration_seconds": time.monotonic() - started,
                "stdout": completed.stdout[-2000:],
                "stderr": completed.stderr[-2000:],
            }
        except subprocess.TimeoutExpired as error:
            stdout = error.stdout.decode(errors="replace") if isinstance(error.stdout, bytes) else error.stdout
            stderr = error.stderr.decode(errors="replace") if isinstance(error.stderr, bytes) else error.stderr
            return {
                "passed": False,
                "returncode": None,
                "timed_out": True,
                "duration_seconds": time.monotonic() - started,
                "stdout": (stdout or "")[-2000:],
                "stderr": (stderr or "")[-2000:],
            }


def python_minify(
    source: str,
    *,
    executable: Path,
    timeout: float,
    work_root: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    work_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="minify-", dir=work_root) as directory:
        source_path = Path(directory) / "solution.py"
        source_path.write_text(source, encoding="utf-8")
        command = [
            executable.as_posix(),
            "--remove-literal-statements",
            source_path.name,
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=directory,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            return {
                "ok": False,
                "code": None,
                "error": "timeout",
                "stderr": str(error.stderr or "")[-2000:],
                "duration_seconds": time.monotonic() - started,
            }
        return {
            "ok": completed.returncode == 0,
            "code": completed.stdout if completed.returncode == 0 else None,
            "error": None if completed.returncode == 0 else "process_failed",
            "returncode": completed.returncode,
            "stderr": completed.stderr[-2000:],
            "duration_seconds": time.monotonic() - started,
        }


def expected_shard_bounds(shard_index: int, shard_count: int) -> tuple[int, int]:
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("invalid shard index/count")
    size = math.ceil(EXPECTED_TASKS / shard_count)
    return shard_index * size, min(EXPECTED_TASKS, (shard_index + 1) * size)


def run_shard(args: argparse.Namespace) -> None:
    shard_index = args.shard_index
    if shard_index is None:
        value = os.environ.get("SLURM_ARRAY_TASK_ID")
        if value is None:
            raise ValueError("--shard-index or SLURM_ARRAY_TASK_ID is required")
        shard_index = int(value)
    start, end = expected_shard_bounds(shard_index, args.shard_count)
    output_path = args.output_root / "shards" / f"part-{shard_index:03d}.jsonl"
    expected_rows = (end - start) * EXPECTED_COMPLETIONS
    if output_path.is_file() and not args.overwrite:
        existing = read_jsonl(output_path)
        if len(existing) == expected_rows:
            print(
                json.dumps(
                    {
                        "status": "already_complete",
                        "shard_index": shard_index,
                        "task_range": [start, end],
                        "rows": len(existing),
                        "output": output_path.as_posix(),
                    }
                )
            )
            return
        raise ValueError(
            f"{output_path} exists with {len(existing)} rows; use --overwrite"
        )

    executable = args.env_bin / "pyminify"
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise FileNotFoundError(f"Python-Minifier executable unavailable: {executable}")

    tasks = load_task_inputs(args.source_results)
    work_root = args.output_root / "work" / f"part-{shard_index:03d}"
    rows: list[dict[str, Any]] = []
    for task in tasks[start:end]:
        for completion_index, source in enumerate(task["generations"]):
            transformed = python_minify(
                source,
                executable=executable,
                timeout=args.transform_timeout,
                work_root=work_root,
            )
            execution = None
            if transformed["ok"]:
                execution = execute_humaneval(
                    transformed["code"],
                    task["test"],
                    task["entry_point"],
                    timeout=args.execution_timeout,
                    memory_mb=args.memory_mb,
                    work_root=work_root,
                )
            rows.append(
                {
                    "task_index": task["task_index"],
                    "source_task_id": task["source_task_id"],
                    "completion_index": completion_index,
                    "reconstructed_empty_task": task["reconstructed_empty"],
                    "baseline_passed": task["baseline_passed"][completion_index],
                    "source_code": source,
                    "python_minifier": transformed,
                    "attacked_passed": bool(execution and execution["passed"]),
                    "execution": execution,
                }
            )

    write_jsonl_atomic(output_path, rows)
    print(
        json.dumps(
            {
                "status": "complete",
                "shard_index": shard_index,
                "task_range": [start, end],
                "rows": len(rows),
                "reconstructed_empty_tasks": sorted(
                    {
                        row["task_index"]
                        for row in rows
                        if row["reconstructed_empty_task"]
                    }
                ),
                "output": output_path.as_posix(),
            }
        )
    )


def pass_at_k(flags: list[bool], k: int) -> float:
    n = len(flags)
    c = sum(flags)
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def aggregate(args: argparse.Namespace) -> None:
    rows: list[dict[str, Any]] = []
    for shard_index in range(args.shard_count):
        path = args.output_root / "shards" / f"part-{shard_index:03d}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"missing CPU shard: {path}")
        rows.extend(read_jsonl(path))

    reconstructed_indices = sorted(
        {
            int(row["task_index"])
            for row in rows
            if row["reconstructed_empty_task"]
        }
    )
    if args.exclude_reconstructed_empty:
        rows = [row for row in rows if not row["reconstructed_empty_task"]]
    task_indices = sorted({int(row["task_index"]) for row in rows})
    task_count = len(task_indices)
    expected = task_count * EXPECTED_COMPLETIONS
    if len(rows) != expected:
        raise ValueError(f"expected {expected} rows, found {len(rows)}")
    keys = {(row["task_index"], row["completion_index"]) for row in rows}
    if len(keys) != expected:
        raise ValueError("duplicate or missing task/completion pairs")

    by_task: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[int(row["task_index"])].append(row)
    if set(by_task) != set(task_indices):
        raise ValueError("task coverage does not match the selected cohort")

    baseline_pass_1 = []
    baseline_pass_10 = []
    attacked_pass_1 = []
    attacked_pass_10 = []
    for task_index in task_indices:
        task_rows = sorted(
            by_task[task_index], key=lambda row: row["completion_index"]
        )
        baseline = [bool(row["baseline_passed"]) for row in task_rows]
        attacked = [bool(row["attacked_passed"]) for row in task_rows]
        baseline_pass_1.append(pass_at_k(baseline, 1))
        baseline_pass_10.append(pass_at_k(baseline, 10))
        attacked_pass_1.append(pass_at_k(attacked, 1))
        attacked_pass_10.append(pass_at_k(attacked, 10))

    metrics = {
        "dataset": "HumanEval",
        "cohort": (
            "official_retained_160"
            if args.exclude_reconstructed_empty
            else "inclusive_164"
        ),
        "task_count": task_count,
        "task_indices": task_indices,
        "completions_per_task": EXPECTED_COMPLETIONS,
        "completion_count": expected,
        "excluded_reconstructed_empty_task_indices": (
            reconstructed_indices if args.exclude_reconstructed_empty else []
        ),
        "reconstructed_empty_task_indices": (
            [] if args.exclude_reconstructed_empty else reconstructed_indices
        ),
        "python_minifier_success_count": sum(
            bool(row["python_minifier"]["ok"]) for row in rows
        ),
        "baseline_correct_completions": sum(
            bool(row["baseline_passed"]) for row in rows
        ),
        "attacked_correct_completions": sum(
            bool(row["attacked_passed"]) for row in rows
        ),
        "baseline_pass_at_1": sum(baseline_pass_1) / task_count,
        "baseline_pass_at_10": sum(baseline_pass_10) / task_count,
        "attacked_pass_at_1": sum(attacked_pass_1) / task_count,
        "attacked_pass_at_10": sum(attacked_pass_10) / task_count,
        "baseline_passes_preserved": sum(
            bool(row["baseline_passed"]) and bool(row["attacked_passed"])
            for row in rows
        ),
        "baseline_pass_regressions": sum(
            bool(row["baseline_passed"]) and not bool(row["attacked_passed"])
            for row in rows
        ),
        "baseline_fail_improvements": sum(
            not bool(row["baseline_passed"]) and bool(row["attacked_passed"])
            for row in rows
        ),
    }
    output = args.output_root / args.output_name
    write_json_atomic(output, metrics)
    print(json.dumps(metrics, indent=2))


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    subparsers = value.add_subparsers(dest="command", required=True)

    shard = subparsers.add_parser("shard")
    shard.add_argument("--source-results", type=Path, required=True)
    shard.add_argument("--output-root", type=Path, required=True)
    shard.add_argument("--shard-index", type=int)
    shard.add_argument("--shard-count", type=int, default=4)
    shard.add_argument(
        "--env-bin",
        type=Path,
        default=Path.home() / "conda" / "envs" / "watermarking" / "bin",
    )
    shard.add_argument("--transform-timeout", type=float, default=10.0)
    shard.add_argument("--execution-timeout", type=float, default=3.0)
    shard.add_argument("--memory-mb", type=int, default=1024)
    shard.add_argument("--overwrite", action="store_true")
    shard.set_defaults(function=run_shard)

    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--output-root", type=Path, required=True)
    aggregate_parser.add_argument("--shard-count", type=int, default=4)
    aggregate_parser.add_argument("--exclude-reconstructed-empty", action="store_true")
    aggregate_parser.add_argument("--output-name", default="metrics.json")
    aggregate_parser.set_defaults(function=aggregate)
    return value


def main() -> None:
    args = parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
