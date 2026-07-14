"""Differential HumanEval/MBPP corpus runner.

The runner first executes the saved baseline solution and test in a dedicated
subprocess. Only a baseline that passes in the current environment is walked;
the exact same test is then executed against the obfuscated source. All output
and temporary working directories are constrained to ``new_prototype/data``.

This is process isolation with resource limits, not a security sandbox. Do not
run untrusted code from unknown sources on a sensitive host.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Iterable, Iterator, Mapping

from .engine import RandomWalkObfuscator


HERE = Path(__file__).resolve().parents[1]
DATA_ROOT = HERE / "data"
REPO_ROOT = HERE.parents[1]
RESULT_ROOT = REPO_ROOT / "data" / "result"
ORIGINAL_ROOT = REPO_ROOT / "data" / "original"

# This is the Python prelude used by the repository's original evaluator.
# NumPy is optional here so the standard HumanEval/MBPP references do not fail
# merely because that unrelated third-party package is absent.
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


@dataclass(frozen=True)
class CorpusCase:
    dataset: str
    task_name: str
    source: str
    test: str
    entry_point: str | None
    origin: str
    stored_passed: bool | None


@dataclass(frozen=True)
class ExecutionResult:
    passed: bool
    returncode: int | None
    timed_out: bool
    duration_seconds: float
    stdout: str
    stderr: str

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "returncode": self.returncode,
            "timed_out": self.timed_out,
            "duration_seconds": self.duration_seconds,
            "stdout": self.stdout,
            "stderr": self.stderr,
        }


def _jsonl(path: Path) -> Iterator[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            yield value


def discover_model_files() -> tuple[Path, ...]:
    files = [
        path
        for path in RESULT_ROOT.glob("*--*_py/*/generate.jsonl")
        if "humaneval_py" in path.parts[-3] or "mbpp_py" in path.parts[-3]
    ]
    return tuple(sorted(files))


def model_cases(
    files: Iterable[Path], *, require_stored_pass: bool = False
) -> Iterator[CorpusCase]:
    for path in files:
        dataset = "humaneval_py" if "humaneval_py" in path.parts[-3] else "mbpp_py"
        for row in _jsonl(path):
            source = row.get("solution")
            test = row.get("test")
            if not isinstance(source, str) or not isinstance(test, str):
                continue
            stored = row.get("passed")
            stored_passed = stored if isinstance(stored, bool) else None
            if require_stored_pass and stored_passed is not True:
                continue
            yield CorpusCase(
                dataset=dataset,
                task_name=str(row.get("task_name", row.get("id", "unknown"))),
                source=source,
                test=test,
                entry_point=(
                    row.get("entry_point")
                    if isinstance(row.get("entry_point"), str)
                    else None
                ),
                origin=path.relative_to(REPO_ROOT).as_posix(),
                stored_passed=stored_passed,
            )


def reference_cases() -> Iterator[CorpusCase]:
    human_eval = ORIGINAL_ROOT / "humaneval-x_py.jsonl"
    if human_eval.is_file():
        for row in _jsonl(human_eval):
            prompt = row.get("prompt")
            solution = row.get("canonical_solution")
            test = row.get("test")
            if isinstance(prompt, str) and isinstance(solution, str) and isinstance(test, str):
                yield CorpusCase(
                    dataset="humaneval_py",
                    task_name=str(row.get("task_id", "unknown")),
                    source=prompt + solution,
                    test=test,
                    entry_point=(
                        row.get("entry_point")
                        if isinstance(row.get("entry_point"), str)
                        else None
                    ),
                    origin=human_eval.relative_to(REPO_ROOT).as_posix(),
                    stored_passed=None,
                )

    mbpp = ORIGINAL_ROOT / "mbppp_py.jsonl"
    if mbpp.is_file():
        for row in _jsonl(mbpp):
            solution = row.get("canonical_solution")
            assertion = row.get("assertion")
            if isinstance(assertion, list):
                assertion = "\n".join(str(item) for item in assertion)
            if isinstance(solution, str) and isinstance(assertion, str):
                yield CorpusCase(
                    dataset="mbpp_py",
                    task_name=str(row.get("task_id", "unknown")),
                    source=solution,
                    test=assertion,
                    entry_point=(
                        row.get("entry_point")
                        if isinstance(row.get("entry_point"), str)
                        else None
                    ),
                    origin=mbpp.relative_to(REPO_ROOT).as_posix(),
                    stored_passed=None,
                )


def _limit_resources(timeout: float, memory_mb: int) -> None:
    import resource

    cpu_seconds = max(1, int(math.ceil(timeout)) + 1)
    memory_bytes = memory_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    resource.setrlimit(resource.RLIMIT_FSIZE, (8 * 1024 * 1024, 8 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))


def execute_case(
    source: str,
    test: str,
    *,
    timeout: float,
    memory_mb: int,
    hash_seed: int = 0,
) -> ExecutionResult:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    execution_root = DATA_ROOT / "execution"
    execution_root.mkdir(parents=True, exist_ok=True)
    script = (
        PYTHON_IMPORT_HELPER.rstrip()
        + "\n\n"
        + source.rstrip()
        + "\n\n"
        + test.lstrip()
        + "\n"
    )
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="case-", dir=execution_root) as directory:
        try:
            completed = subprocess.run(
                # ``-I`` implies ``-E`` and therefore silently ignored the
                # PYTHONHASHSEED supplied below.  ``-s`` still disables the
                # user site while allowing the deliberately minimal child
                # environment to make hash-dependent tests deterministic.
                [sys.executable, "-s", "-c", script],
                cwd=directory,
                env={
                    "PATH": os.environ.get("PATH", ""),
                    "PYTHONHASHSEED": str(int(hash_seed)),
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "NUMEXPR_NUM_THREADS": "1",
                    "BLIS_NUM_THREADS": "1",
                    "VECLIB_MAXIMUM_THREADS": "1",
                },
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                preexec_fn=lambda: _limit_resources(timeout, memory_mb),
            )
            return ExecutionResult(
                passed=completed.returncode == 0,
                returncode=completed.returncode,
                timed_out=False,
                duration_seconds=time.monotonic() - started,
                stdout=completed.stdout[-4000:],
                stderr=completed.stderr[-4000:],
            )
        except subprocess.TimeoutExpired as error:
            return ExecutionResult(
                passed=False,
                returncode=None,
                timed_out=True,
                duration_seconds=time.monotonic() - started,
                stdout=(error.stdout or "")[-4000:],
                stderr=(error.stderr or "")[-4000:],
            )


def _output_path(value: str) -> Path:
    path = (DATA_ROOT / value).resolve()
    path.relative_to(DATA_ROOT.resolve())
    return path


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
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
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def evaluate(
    cases: Iterable[CorpusCase],
    *,
    seeds: tuple[int, ...],
    steps: tuple[int, ...],
    timeout: float,
    memory_mb: int,
    max_cases: int | None,
) -> Iterator[dict[str, object]]:
    for case_index, case in enumerate(cases):
        if max_cases is not None and case_index >= max_cases:
            break
        baseline = execute_case(
            case.source, case.test, timeout=timeout, memory_mb=memory_mb
        )
        common: dict[str, object] = {
            "case_index": case_index,
            "dataset": case.dataset,
            "task_name": case.task_name,
            "entry_point": case.entry_point,
            "origin": case.origin,
            "stored_passed": case.stored_passed,
            "baseline": baseline.to_dict(),
        }
        if not baseline.passed:
            yield {**common, "status": "baseline_failed"}
            continue
        for seed in seeds:
            for step_count in steps:
                try:
                    engine = RandomWalkObfuscator(case.source, seed=seed)
                    walked = engine.walk(case.source, step_count)
                    obfuscated = execute_case(
                        walked.source,
                        case.test,
                        timeout=timeout,
                        memory_mb=memory_mb,
                    )
                    yield {
                        **common,
                        "status": (
                            "preserved" if obfuscated.passed else "regression"
                        ),
                        "seed": seed,
                        "steps": step_count,
                        "rule_counts": walked.rule_counts,
                        "trace": [record.to_dict() for record in walked.records],
                        "source_bytes": len(case.source.encode("utf-8")),
                        "obfuscated_bytes": len(walked.source.encode("utf-8")),
                        "obfuscated": obfuscated.to_dict(),
                    }
                except BaseException as error:
                    yield {
                        **common,
                        "status": "obfuscator_error",
                        "seed": seed,
                        "steps": step_count,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }


def _csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(piece) for piece in value.split(",") if piece.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="*",
        help="generate.jsonl files; default discovers all Python result files",
    )
    parser.add_argument("--include-reference", action="store_true")
    parser.add_argument("--require-stored-pass", action="store_true")
    parser.add_argument("--seeds", type=_csv_ints, default=(0, 1, 2))
    parser.add_argument("--steps", type=_csv_ints, default=(1, 5, 10, 25, 50))
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--memory-mb", type=int, default=1024)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--output", default="corpus_results.jsonl")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    files = tuple(args.inputs) if args.inputs else discover_model_files()
    cases: Iterable[CorpusCase] = model_cases(
        files, require_stored_pass=args.require_stored_pass
    )
    if args.include_reference:
        import itertools

        cases = itertools.chain(reference_cases(), cases)
    rows = evaluate(
        cases,
        seeds=args.seeds,
        steps=args.steps,
        timeout=args.timeout,
        memory_mb=args.memory_mb,
        max_cases=args.max_cases,
    )
    _write_jsonl(_output_path(args.output), rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
