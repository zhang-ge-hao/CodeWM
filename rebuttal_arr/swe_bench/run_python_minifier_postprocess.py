#!/usr/bin/env python3
"""Post-process completed SWE-bench patches with Python-Minifier.

The runner deliberately reuses completed model patches.  It reconstructs the
official base commit, applies one patch, minifies only functions or methods
whose post-image lines are touched by that patch, rebuilds a diff against the
same base commit, detects WLLM on all added lines of the final diff, and
optionally invokes the official Modal SWE-bench evaluator.
"""

from __future__ import annotations

import argparse
import ast
import fcntl
import hashlib
import json
import logging
import os
import re
import shutil
import statistics
import subprocess
import tempfile
import textwrap
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

from run_experiment import (
    aggregate_official_reports,
    atomic_write_json,
    atomic_write_jsonl,
    configure_logging,
    evaluate_one_modal,
    load_evaluation_statuses,
    prepare_async_evaluation,
)
from run_mini_experiment import (
    NGRAM_LEN,
    create_patch_detector,
    detect_added_code,
    extract_added_code,
)


VERIFIED_DATASET = "SWE-bench/SWE-bench_Verified"
OBFUSCATOR_NAME = "Python-Minifier"
OBFUSCATOR_VERSION = "2.11.3"
OBFUSCATOR_ARGUMENTS = ("--remove-literal-statements",)
SCHEMA_VERSION = 3
LOGGER = logging.getLogger("swe_bench_experiment")
REPO_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
HUNK_PATTERN = re.compile(
    r"^@@ -\d+(?:,\d+)? \+(?P<start>\d+)(?:,(?P<count>\d+))? @@"
)


@dataclass(frozen=True, order=True)
class FunctionSpan:
    start_line: int
    end_line: int
    qualified_name: str


@dataclass(frozen=True)
class TransformResult:
    status: str
    final_patch: str
    candidate_python_paths: tuple[str, ...]
    transformed_python_paths: tuple[str, ...]
    transformed_python_functions: tuple[str, ...]
    obfuscator_seconds: float
    obfuscator_invocations: int
    error: str | None
    elapsed_seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--repo-cache", type=Path, required=True)
    parser.add_argument("--pyminify", type=Path, required=True)
    parser.add_argument("--swebench-python", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--transform-workers", type=int, default=4)
    parser.add_argument("--eval-workers", type=int, default=4)
    parser.add_argument("--modal-eval-timeout", type=int, default=1800)
    parser.add_argument(
        "--instance-id",
        action="append",
        default=[],
        help="Restrict the run to one instance. May be repeated.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--skip-detection",
        action="store_true",
        help="Transform and compare patches without loading or running WLLM detection.",
    )
    parser.add_argument("--no-reuse-identical-evaluations", action="store_true")
    args = parser.parse_args()

    if args.source_dir.resolve() == args.data_dir.resolve():
        parser.error("--source-dir and --data-dir must differ")
    if args.transform_workers <= 0 or args.eval_workers <= 0:
        parser.error("worker counts must be positive")
    if args.modal_eval_timeout <= 0:
        parser.error("--modal-eval-timeout must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    for path, label in (
        (args.source_dir, "source directory"),
        (args.pyminify, "pyminify executable"),
        (args.swebench_python, "SWE-bench Python"),
    ):
        if not path.exists():
            parser.error(f"{label} does not exist: {path}")
    return args


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON at {path}:{line_number}") from exc
    return rows


def run_command(
    command: list[str],
    *,
    cwd: Path | None = None,
    timeout: float = 600.0,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["GIT_TERMINAL_PROMPT"] = "0"
    result = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )
    if check and result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        detail = stderr or stdout or "no command output"
        raise RuntimeError(
            f"Command failed ({result.returncode}): {command!r}: {detail[-4000:]}"
        )
    return result


def python_paths_from_patch(patch: str) -> tuple[str, ...]:
    """Return candidate post-image Python paths from unified diff headers."""

    paths: list[str] = []
    for line in patch.splitlines():
        if not line.startswith("+++ "):
            continue
        value = line[4:].split("\t", 1)[0]
        if value == "/dev/null":
            continue
        if value.startswith("b/"):
            value = value[2:]
        if value.endswith(".py") and value not in paths:
            paths.append(value)
    return tuple(paths)


def all_paths_from_patch(patch: str) -> tuple[str, ...]:
    """Return pre- and post-image paths needed to apply one unified diff."""

    paths: list[str] = []
    for line in patch.splitlines():
        if not line.startswith(("--- ", "+++ ")):
            continue
        value = line[4:].split("\t", 1)[0]
        if value == "/dev/null":
            continue
        if value.startswith(("a/", "b/")):
            value = value[2:]
        if value and value not in paths:
            paths.append(value)
    return tuple(paths)


def changed_post_image_lines(zero_context_patch: str) -> set[int]:
    """Return added/replacement line numbers from a zero-context file diff.

    A deletion-only hunk has no surviving post-image code to transform and is
    deliberately ignored.  Replacements and insertions have a positive
    post-image count and therefore identify the containing function exactly.
    """

    changed: set[int] = set()
    for line in zero_context_patch.splitlines():
        match = HUNK_PATTERN.match(line)
        if not match:
            continue
        start = int(match.group("start"))
        count = int(match.group("count") or "1")
        changed.update(range(start, start + count))
    return changed


class _FunctionSpanVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.scope: list[str] = []
        self.spans: list[FunctionSpan] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def _visit_function(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> None:
        decorator_lines = [value.lineno for value in node.decorator_list]
        start = min([node.lineno, *decorator_lines])
        end = node.end_lineno
        if end is None:
            raise RuntimeError(f"AST lacks end line for function {node.name!r}")
        qualified_name = ".".join([*self.scope, node.name])
        self.spans.append(FunctionSpan(start, end, qualified_name))
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)


def modified_function_spans(source: str, changed_lines: set[int]) -> list[FunctionSpan]:
    """Find the smallest surviving function containing each changed line."""

    tree = ast.parse(source)
    visitor = _FunctionSpanVisitor()
    visitor.visit(tree)
    selected: set[FunctionSpan] = set()
    for line_number in changed_lines:
        containing = [
            span
            for span in visitor.spans
            if span.start_line <= line_number <= span.end_line
        ]
        if containing:
            selected.add(
                min(
                    containing,
                    key=lambda value: (
                        value.end_line - value.start_line,
                        -value.start_line,
                    ),
                )
            )

    # If separate edits select an outer function and one of its nested
    # functions, transforming the outer span already covers the inner one.
    non_overlapping: list[FunctionSpan] = []
    for span in sorted(selected, key=lambda value: (value.start_line, -value.end_line)):
        if any(
            outer.start_line <= span.start_line and outer.end_line >= span.end_line
            for outer in non_overlapping
        ):
            continue
        non_overlapping.append(span)
    return non_overlapping


def reindent_minified_function(minified: str, indentation: str, had_newline: bool) -> str:
    value = minified.rstrip("\r\n")
    value = textwrap.indent(value, indentation, predicate=lambda _line: True)
    if had_newline:
        value += "\n"
    return value


def added_line_metrics(original_patch: str, final_patch: str) -> dict[str, Any]:
    """Compare exactly the added code lines passed to the WLLM detector.

    Lines are matched as a multiset, so duplicate code lines contribute at
    most their minimum count in the two patches.  Ordering is intentionally
    ignored, matching the standard line-level precision/recall/F1 definition.
    """

    original_lines = extract_added_code(original_patch).splitlines()
    final_lines = extract_added_code(final_patch).splitlines()
    overlap = sum((Counter(original_lines) & Counter(final_lines)).values())
    if final_lines:
        precision = overlap / len(final_lines)
    else:
        precision = 1.0 if not original_lines else 0.0
    if original_lines:
        recall = overlap / len(original_lines)
    else:
        recall = 1.0 if not final_lines else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "original_added_lines": len(original_lines),
        "final_added_lines": len(final_lines),
        "overlapping_added_lines": overlap,
        "added_line_precision": precision,
        "added_line_recall": recall,
        "added_line_f1": f1,
    }


def _cache_name(repo: str) -> str:
    if not REPO_PATTERN.fullmatch(repo):
        raise RuntimeError(f"Unsafe repository identifier: {repo!r}")
    return repo.replace("/", "__") + ".git"


def ensure_repo_mirror(repo: str, base_commit: str, cache_root: Path) -> Path:
    cache_root.mkdir(parents=True, exist_ok=True)
    mirror = cache_root / _cache_name(repo)
    lock_path = cache_root / (_cache_name(repo) + ".lock")
    lock_path.touch(exist_ok=True)
    with lock_path.open("r+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if not mirror.exists():
            temporary = cache_root / (mirror.name + f".tmp-{os.getpid()}")
            if temporary.exists():
                raise RuntimeError(f"Stale temporary mirror exists: {temporary}")
            run_command(
                [
                    "git",
                    "clone",
                    "--mirror",
                    "--filter=blob:none",
                    f"https://github.com/{repo}.git",
                    str(temporary),
                ],
                timeout=3600.0,
            )
            temporary.replace(mirror)
        probe = run_command(
            ["git", "--git-dir", str(mirror), "cat-file", "-e", f"{base_commit}^{{commit}}"],
            check=False,
        )
        if probe.returncode != 0:
            run_command(
                [
                    "git",
                    "--git-dir",
                    str(mirror),
                    "fetch",
                    "--filter=blob:none",
                    "origin",
                    base_commit,
                ],
                timeout=1800.0,
            )
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return mirror


def minify_staged_python_patch(
    *,
    repo: str,
    base_commit: str,
    original_patch: str,
    cache_root: Path,
    pyminify: Path,
    work_root: Path,
) -> TransformResult:
    started = time.monotonic()
    obfuscator_seconds = 0.0
    obfuscator_invocations = 0
    candidates = python_paths_from_patch(original_patch)
    touched_paths = all_paths_from_patch(original_patch)
    if not original_patch.strip():
        return TransformResult(
            status="empty_patch",
            final_patch=original_patch,
            candidate_python_paths=(),
            transformed_python_paths=(),
            transformed_python_functions=(),
            obfuscator_seconds=0.0,
            obfuscator_invocations=0,
            error=None,
            elapsed_seconds=round(time.monotonic() - started, 3),
        )
    if not candidates:
        return TransformResult(
            status="no_python_changes",
            final_patch=original_patch,
            candidate_python_paths=(),
            transformed_python_paths=(),
            transformed_python_functions=(),
            obfuscator_seconds=0.0,
            obfuscator_invocations=0,
            error=None,
            elapsed_seconds=round(time.monotonic() - started, 3),
        )

    try:
        mirror = ensure_repo_mirror(repo, base_commit, cache_root)
        work_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="python-minifier-swebench-", dir=work_root
        ) as temporary:
            checkout = Path(temporary) / "repo"
            run_command(
                ["git", "clone", "--shared", "--no-checkout", str(mirror), str(checkout)],
                timeout=1800.0,
            )
            # A shared clone of a blob-filtered mirror cannot ask that mirror
            # for promised blobs it does not yet own.  Point the promisor
            # remote back to the public source and sparsely materialize only
            # files involved in this patch.
            run_command(
                [
                    "git",
                    "remote",
                    "set-url",
                    "origin",
                    f"https://github.com/{repo}.git",
                ],
                cwd=checkout,
            )
            run_command(
                ["git", "config", "remote.origin.promisor", "true"], cwd=checkout
            )
            run_command(
                ["git", "config", "remote.origin.partialclonefilter", "blob:none"],
                cwd=checkout,
            )
            if touched_paths:
                run_command(
                    ["git", "sparse-checkout", "init", "--no-cone"], cwd=checkout
                )
                run_command(
                    [
                        "git",
                        "sparse-checkout",
                        "set",
                        "--no-cone",
                        *touched_paths,
                    ],
                    cwd=checkout,
                )
            run_command(["git", "checkout", "--detach", base_commit], cwd=checkout)
            patch_path = Path(temporary) / "original.patch"
            patch_path.write_text(original_patch)
            run_command(
                [
                    "git",
                    "apply",
                    "--index",
                    "--whitespace=nowarn",
                    str(patch_path.resolve()),
                ],
                cwd=checkout,
            )
            names = run_command(
                [
                    "git",
                    "diff",
                    "--cached",
                    "--name-only",
                    "--diff-filter=AM",
                    "-z",
                    "HEAD",
                ],
                cwd=checkout,
            ).stdout
            staged_paths = tuple(value for value in names.split("\0") if value)
            python_paths = tuple(
                value
                for value in staged_paths
                if value.endswith(".py") and (checkout / value).is_file()
            )
            if not python_paths:
                return TransformResult(
                    status="no_python_changes",
                    final_patch=original_patch,
                    candidate_python_paths=candidates,
                    transformed_python_paths=(),
                    transformed_python_functions=(),
                    obfuscator_seconds=0.0,
                    obfuscator_invocations=0,
                    error=None,
                    elapsed_seconds=round(time.monotonic() - started, 3),
                )

            transformed_paths: list[str] = []
            transformed_functions: list[str] = []
            snippet_root = Path(temporary) / "snippets"
            snippet_root.mkdir()
            for file_index, relative in enumerate(python_paths, 1):
                source_path = checkout / relative
                mode = source_path.stat().st_mode
                zero_context_patch = run_command(
                    [
                        "git",
                        "diff",
                        "--cached",
                        "--unified=0",
                        "--no-color",
                        "HEAD",
                        "--",
                        relative,
                    ],
                    cwd=checkout,
                ).stdout
                changed_lines = changed_post_image_lines(zero_context_patch)
                source = source_path.read_text()
                spans = modified_function_spans(source, changed_lines)
                if not spans:
                    continue
                source_lines = source.splitlines(keepends=True)
                replacements: list[tuple[FunctionSpan, str]] = []
                for function_index, span in enumerate(spans, 1):
                    original_function = "".join(
                        source_lines[span.start_line - 1 : span.end_line]
                    )
                    first_line = original_function.splitlines()[0]
                    indentation = first_line[: len(first_line) - len(first_line.lstrip())]
                    dedented = textwrap.dedent(original_function)
                    snippet_path = (
                        snippet_root / f"{file_index:03d}-{function_index:03d}.py"
                    )
                    snippet_path.write_text(dedented)
                    obfuscator_invocations += 1
                    obfuscator_started = time.monotonic()
                    try:
                        result = run_command(
                            [
                                str(pyminify),
                                *OBFUSCATOR_ARGUMENTS,
                                str(snippet_path.resolve()),
                            ],
                            cwd=checkout,
                            timeout=120.0,
                        )
                    finally:
                        obfuscator_seconds += time.monotonic() - obfuscator_started
                    replacements.append(
                        (
                            span,
                            reindent_minified_function(
                                result.stdout,
                                indentation,
                                original_function.endswith(("\n", "\r")),
                            ),
                        )
                    )
                    transformed_functions.append(
                        f"{relative}:{span.qualified_name}@{span.start_line}-{span.end_line}"
                    )
                for span, replacement in reversed(replacements):
                    source_lines[span.start_line - 1 : span.end_line] = [replacement]
                transformed_source = "".join(source_lines)
                # Catch method-extraction/reindentation errors before building
                # a patch that cannot even be parsed as Python.
                ast.parse(transformed_source, filename=relative)
                source_path.write_text(transformed_source)
                source_path.chmod(mode)
                transformed_paths.append(relative)

            if not transformed_functions:
                return TransformResult(
                    status="no_modified_functions",
                    final_patch=original_patch,
                    candidate_python_paths=candidates,
                    transformed_python_paths=(),
                    transformed_python_functions=(),
                    obfuscator_seconds=0.0,
                    obfuscator_invocations=0,
                    error=None,
                    elapsed_seconds=round(time.monotonic() - started, 3),
                )
            run_command(["git", "add", "-A"], cwd=checkout)
            final_patch = run_command(
                [
                    "git",
                    "diff",
                    "--cached",
                    "--binary",
                    "--full-index",
                    "HEAD",
                ],
                cwd=checkout,
            ).stdout
            if not final_patch.strip():
                raise RuntimeError("Python-Minifier produced an empty final patch")
            return TransformResult(
                status="transformed",
                final_patch=final_patch,
                candidate_python_paths=candidates,
                transformed_python_paths=tuple(transformed_paths),
                transformed_python_functions=tuple(transformed_functions),
                obfuscator_seconds=round(obfuscator_seconds, 6),
                obfuscator_invocations=obfuscator_invocations,
                error=None,
                elapsed_seconds=round(time.monotonic() - started, 3),
            )
    except Exception as exc:
        return TransformResult(
            status="transform_failed",
            final_patch=original_patch,
            candidate_python_paths=candidates,
            transformed_python_paths=(),
            transformed_python_functions=(),
            obfuscator_seconds=round(obfuscator_seconds, 6),
            obfuscator_invocations=obfuscator_invocations,
            error=f"{type(exc).__name__}: {exc}",
            elapsed_seconds=round(time.monotonic() - started, 3),
        )


def prediction_name(source_row: dict[str, Any]) -> str:
    return source_row["model_name_or_path"] + "--python-minifier-method-scope-2.11.3"


def prepare_case(
    *,
    index: int,
    case: dict[str, Any],
    source_row: dict[str, Any],
    data_dir: Path,
    cache_root: Path,
    pyminify: Path,
    work_root: Path,
    detector,
    tokenizer,
    detector_args: SimpleNamespace,
) -> dict[str, Any]:
    instance_id = case["instance_id"]
    case_dir = data_dir / "cases" / f"case-{index:03d}-{instance_id}"
    result_path = case_dir / "result.json"
    original_patch = source_row.get("model_patch") or ""
    original_sha = sha256_text(original_patch)
    if result_path.exists():
        existing = json.loads(result_path.read_text())
        if (
            existing.get("schema_version") == SCHEMA_VERSION
            and existing.get("original_patch_sha256") == original_sha
        ):
            return existing

    case_dir.mkdir(parents=True, exist_ok=True)
    transform = minify_staged_python_patch(
        repo=case["repo"],
        base_commit=case["base_commit"],
        original_patch=original_patch,
        cache_root=cache_root,
        pyminify=pyminify,
        work_root=work_root,
    )
    case_detector = (
        create_patch_detector(tokenizer, detector_args)
        if detector is not None
        else None
    )
    added_code, post_detection = detect_added_code(
        transform.final_patch,
        case_detector,
        tokenizer,
        detector_args,
        # Each worker owns its detector and RNG, so independent cases can be
        # scored concurrently without changing the greenlist algorithm.
        serialize_detector=False,
    )
    (case_dir / "original_patch.diff").write_text(original_patch)
    (case_dir / "obfuscated_patch.diff").write_text(transform.final_patch)
    (case_dir / "added_code_pre.txt").write_text(extract_added_code(original_patch))
    (case_dir / "added_code_post.txt").write_text(added_code)
    line_metrics = added_line_metrics(original_patch, transform.final_patch)
    row = {
        "schema_version": SCHEMA_VERSION,
        "instance_id": instance_id,
        "selection_index": index,
        "repo": case["repo"],
        "base_commit": case["base_commit"],
        "model_name_or_path": prediction_name(source_row),
        "model_patch": transform.final_patch,
        "original_patch_sha256": original_sha,
        "final_patch_sha256": sha256_text(transform.final_patch),
        "original_patch_characters": len(original_patch),
        "final_patch_characters": len(transform.final_patch),
        **line_metrics,
        "patch_identical": transform.final_patch == original_patch,
        "watermarking": detector_args.watermarking,
        "delta": detector_args.delta,
        "gamma": detector_args.gamma,
        "ngram_len": NGRAM_LEN,
        "pre_detection": source_row.get("detection"),
        "post_detection": post_detection,
        # Keep the established field name for shared summarization/debug tools.
        "detection": post_detection,
        "obfuscation": asdict(transform),
    }
    # Avoid duplicating the potentially large patch in the nested metadata.
    row["obfuscation"].pop("final_patch", None)
    atomic_write_json(result_path, row)
    LOGGER.info(
        "[%d] Post-process %s: status=%s files=%d functions=%d added-lines=%d->%d precision=%.3f recall=%.3f f1=%.3f obfuscator=%.3fs end-to-end=%.3fs z=%s",
        index,
        instance_id,
        transform.status,
        len(transform.transformed_python_paths),
        len(transform.transformed_python_functions),
        line_metrics["original_added_lines"],
        line_metrics["final_added_lines"],
        line_metrics["added_line_precision"],
        line_metrics["added_line_recall"],
        line_metrics["added_line_f1"],
        transform.obfuscator_seconds,
        transform.elapsed_seconds,
        post_detection.get("z_score") if isinstance(post_detection, dict) else None,
    )
    return row


def copy_reused_evaluation(
    *,
    source_status: dict[str, Any],
    data_dir: Path,
    index: int,
    run_id: str,
    row: dict[str, Any],
) -> None:
    source_case_dir = Path(source_status["_case_dir"])
    source_report = source_case_dir / "official_report.json"
    if not source_status.get("evaluation_complete") or not source_report.is_file():
        raise RuntimeError("Source evaluation is not reusable")
    target_dir = data_dir / "evaluations" / f"case-{index:03d}"
    target_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(
        target_dir / "predictions.jsonl",
        [
            {
                "instance_id": row["instance_id"],
                "model_name_or_path": row["model_name_or_path"],
                "model_patch": row["model_patch"],
            }
        ],
    )
    shutil.copyfile(source_report, target_dir / "official_report.json")
    status = {
        "schema_version": SCHEMA_VERSION,
        "base_run_id": run_id,
        "evaluation_run_id": None,
        "generation_index": index,
        "instance_id": row["instance_id"],
        "prediction_model_name": row["model_name_or_path"],
        "attempt": 0,
        "model_patch_nonempty": bool(row["model_patch"].strip()),
        "evaluation_state": "complete",
        "evaluation_complete": True,
        "outcome": source_status.get("outcome"),
        "reused_evaluation": True,
        "reused_from": str(source_case_dir),
        "official_report_file": "official_report.json",
    }
    atomic_write_json(target_dir / "status.json", status)


def valid_z(detection: Any) -> float | None:
    if not isinstance(detection, dict):
        return None
    if detection.get("invalid") or detection.get("z_score") is None:
        return None
    return float(detection["z_score"])


def normal_auroc(z_scores: Iterable[float]) -> float | None:
    values = list(z_scores)
    if not values:
        return None
    standard_normal = statistics.NormalDist()
    return statistics.fmean(standard_normal.cdf(value) for value in values)


def make_summary(
    *,
    rows: dict[str, dict[str, Any]],
    selected_ids: list[str],
    original_report: dict[str, Any],
    post_report: dict[str, Any] | None,
    settings: dict[str, Any],
) -> dict[str, Any]:
    original_resolved = set(original_report.get("resolved_ids", [])) & set(selected_ids)
    post_resolved = (
        set(post_report.get("resolved_ids", [])) & set(selected_ids)
        if post_report
        else set()
    )
    statuses = {
        instance_id: row["obfuscation"]["status"]
        for instance_id, row in rows.items()
    }
    transformed_rows = [
        row
        for row in rows.values()
        if row["obfuscation"]["status"] == "transformed"
    ]
    # A failed/no-op transformation retains the original patch for audit and
    # optional evaluation reuse.  It must not be counted as a post-attack
    # detection sample.  Use the same successfully transformed population for
    # both pre and post metrics so their AUROCs are directly paired.
    pre_scores = [
        score
        for row in transformed_rows
        if (score := valid_z(row.get("pre_detection"))) is not None
    ]
    post_scores = [
        score
        for row in transformed_rows
        if (score := valid_z(row.get("post_detection"))) is not None
    ]
    paired = [
        (pre, post)
        for row in transformed_rows
        if (pre := valid_z(row.get("pre_detection"))) is not None
        and (post := valid_z(row.get("post_detection"))) is not None
    ]
    nonempty_rows = [
        row for row in rows.values() if row.get("original_patch_characters", 0) > 0
    ]
    timed_rows = [
        row
        for row in transformed_rows
        if row.get("obfuscation", {}).get("obfuscator_seconds") is not None
    ]
    obfuscator_times = [
        float(row["obfuscation"]["obfuscator_seconds"]) for row in timed_rows
    ]
    end_to_end_times = [
        float(row["obfuscation"]["elapsed_seconds"]) for row in timed_rows
    ]
    obfuscator_invocations = sum(
        int(row["obfuscation"]["obfuscator_invocations"]) for row in timed_rows
    )

    def added_line_similarity_counts(
        population: list[dict[str, Any]],
    ) -> dict[str, Any]:
        valid_rows = [
            row for row in population if row.get("added_line_f1") is not None
        ]
        f1_scores = [float(row["added_line_f1"]) for row in valid_rows]
        return {
            "population": len(population),
            "valid_scores": len(f1_scores),
            "f1_eq_1": sum(value >= 1.0 - 1e-12 for value in f1_scores),
            "f1_ge_0.95": sum(value >= 0.95 for value in f1_scores),
            "f1_ge_0.90": sum(value >= 0.90 for value in f1_scores),
            "f1_ge_0.80": sum(value >= 0.80 for value in f1_scores),
            "f1_ge_0.50": sum(value >= 0.50 for value in f1_scores),
            "mean_f1": statistics.fmean(f1_scores) if f1_scores else None,
            "median_f1": statistics.median(f1_scores) if f1_scores else None,
            "mean_precision": (
                statistics.fmean(
                    float(row["added_line_precision"]) for row in valid_rows
                )
                if valid_rows
                else None
            ),
            "mean_recall": (
                statistics.fmean(
                    float(row["added_line_recall"]) for row in valid_rows
                )
                if valid_rows
                else None
            ),
        }
    retained = original_resolved & post_resolved
    return {
        "schema_version": SCHEMA_VERSION,
        "obfuscator": OBFUSCATOR_NAME,
        "obfuscator_version": OBFUSCATOR_VERSION,
        "obfuscator_arguments": list(OBFUSCATOR_ARGUMENTS),
        "selected_cases": len(selected_ids),
        "watermarking": settings["watermarking"],
        "delta": settings.get("delta"),
        "gamma": settings.get("gamma"),
        "ngram_len": NGRAM_LEN,
        "detection_scope": (
            "all_final_patch_added_lines"
            if settings["watermarking"] == "wllm"
            and not settings.get("detection_skipped", False)
            else None
        ),
        "status_counts": {
            status: sum(value == status for value in statuses.values())
            for status in sorted(set(statuses.values()))
        },
        "transformed_cases": sum(value == "transformed" for value in statuses.values()),
        "transform_failed_cases": sum(
            value == "transform_failed" for value in statuses.values()
        ),
        "identical_patch_cases": sum(row["patch_identical"] for row in rows.values()),
        "obfuscation_timing": {
            "timed_transformed_cases": len(timed_rows),
            "obfuscator_invocations": obfuscator_invocations,
            "total_obfuscator_seconds": sum(obfuscator_times),
            "mean_obfuscator_seconds_per_patch": (
                statistics.fmean(obfuscator_times) if obfuscator_times else None
            ),
            "median_obfuscator_seconds_per_patch": (
                statistics.median(obfuscator_times) if obfuscator_times else None
            ),
            "mean_obfuscator_seconds_per_function": (
                sum(obfuscator_times) / obfuscator_invocations
                if obfuscator_invocations
                else None
            ),
            "total_end_to_end_seconds": sum(end_to_end_times),
            "mean_end_to_end_seconds_per_patch": (
                statistics.fmean(end_to_end_times) if end_to_end_times else None
            ),
            "median_end_to_end_seconds_per_patch": (
                statistics.median(end_to_end_times) if end_to_end_times else None
            ),
        },
        "added_line_similarity": {
            "all_selected_cases": added_line_similarity_counts(list(rows.values())),
            "nonempty_patch_cases": added_line_similarity_counts(nonempty_rows),
            "successfully_transformed_cases": added_line_similarity_counts(
                transformed_rows
            ),
        },
        "reused_evaluations": sum(
            row.get("patch_identical", False) for row in rows.values()
        ),
        "original_resolved_cases": len(original_resolved),
        "post_resolved_cases": len(post_resolved) if post_report else None,
        "original_solve_rate": len(original_resolved) / len(selected_ids),
        "post_solve_rate": (
            len(post_resolved) / len(selected_ids) if post_report else None
        ),
        "retained_resolved_cases": len(retained) if post_report else None,
        "retained_resolved_rate": (
            len(retained) / len(original_resolved)
            if post_report and original_resolved
            else None
        ),
        "resolved_to_unresolved_ids": (
            sorted(original_resolved - post_resolved) if post_report else []
        ),
        "unresolved_to_resolved_ids": (
            sorted(post_resolved - original_resolved) if post_report else []
        ),
        "valid_pre_detection_cases": len(pre_scores),
        "valid_post_detection_cases": len(post_scores),
        "paired_detection_cases": len(paired),
        "mean_pre_z_score": statistics.fmean(pre_scores) if pre_scores else None,
        "mean_post_z_score": statistics.fmean(post_scores) if post_scores else None,
        "median_pre_z_score": statistics.median(pre_scores) if pre_scores else None,
        "median_post_z_score": statistics.median(post_scores) if post_scores else None,
        "mean_paired_z_change": (
            statistics.fmean(post - pre for pre, post in paired) if paired else None
        ),
        "pre_auroc_vs_standard_normal": normal_auroc(pre_scores),
        "post_auroc_vs_standard_normal": normal_auroc(post_scores),
    }


def select_cases(
    selection: dict[str, Any], dataset_by_id: dict[str, dict[str, Any]], args: argparse.Namespace
) -> tuple[list[str], list[dict[str, Any]], dict[str, int]]:
    all_ids = list(selection["instance_ids"])
    index_by_id = {instance_id: index for index, instance_id in enumerate(all_ids, 1)}
    if args.instance_id:
        requested = set(args.instance_id)
        missing = requested - set(all_ids)
        if missing:
            raise RuntimeError(f"Requested IDs not in selection: {sorted(missing)}")
        selected_ids = [value for value in all_ids if value in requested]
    else:
        selected_ids = all_ids
    if args.limit is not None:
        selected_ids = selected_ids[: args.limit]
    cases = []
    for instance_id in selected_ids:
        dataset_row = dataset_by_id[instance_id]
        cases.append(
            {
                "instance_id": instance_id,
                "repo": dataset_row["repo"],
                "base_commit": dataset_row["base_commit"],
            }
        )
    return selected_ids, cases, index_by_id


def main() -> None:
    args = parse_args()
    configure_logging(args.data_dir)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    version = run_command([str(args.pyminify), "--version"]).stdout.strip()
    if version != OBFUSCATOR_VERSION:
        raise RuntimeError(
            f"Expected Python-Minifier {OBFUSCATOR_VERSION}, found {version!r}"
        )

    source_settings = json.loads((args.source_dir / "settings.json").read_text())
    selection = json.loads((args.source_dir / "selection.json").read_text())
    source_rows = {
        row["instance_id"]: row
        for row in load_jsonl(args.source_dir / "case_results.jsonl")
    }
    original_report = json.loads(
        (args.source_dir / "aggregate_official_report.json").read_text()
    )
    from datasets import load_dataset

    dataset = load_dataset(VERIFIED_DATASET, split="test")
    dataset_by_id = {row["instance_id"]: dict(row) for row in dataset}
    selected_ids, cases, index_by_id = select_cases(selection, dataset_by_id, args)
    missing_rows = [value for value in selected_ids if value not in source_rows]
    if missing_rows:
        raise RuntimeError(f"Source results missing IDs: {missing_rows}")

    watermarking = source_settings["watermarking"]
    detector_args = SimpleNamespace(
        watermarking=watermarking,
        gamma=source_settings.get("gamma"),
        delta=source_settings.get("delta"),
        watermark_key=source_settings["watermark_key"],
        z_threshold=4.0,
    )
    tokenizer = None
    detector = None
    if watermarking == "wllm" and not args.skip_detection:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            source_settings["model_id"], local_files_only=True
        )
        detector = create_patch_detector(tokenizer, detector_args)

    output_settings = {
        "schema_version": SCHEMA_VERSION,
        "source_dir": str(args.source_dir),
        "source_settings": source_settings,
        "selection": selected_ids,
        "watermarking": watermarking,
        "delta": source_settings.get("delta"),
        "gamma": source_settings.get("gamma"),
        "ngram_len": NGRAM_LEN,
        "watermark_key": source_settings["watermark_key"],
        "detection_scope": (
            "all_final_patch_added_lines"
            if watermarking == "wllm" and not args.skip_detection
            else None
        ),
        "detection_skipped": args.skip_detection,
        "obfuscator": OBFUSCATOR_NAME,
        "obfuscator_version": version,
        "obfuscator_arguments": list(OBFUSCATOR_ARGUMENTS),
        "transformed_scope": "modified_FunctionDef_and_AsyncFunctionDef_only",
        "identity_fallback_on_failure": True,
    }
    atomic_write_json(args.data_dir / "settings.json", output_settings)
    atomic_write_json(
        args.data_dir / "selection.json",
        {
            "schema_version": SCHEMA_VERSION,
            "dataset": VERIFIED_DATASET,
            "instance_ids": selected_ids,
            "source_selection": str(args.source_dir / "selection.json"),
        },
    )

    rows: dict[str, dict[str, Any]] = {}
    work_root = args.data_dir / "work"
    with ThreadPoolExecutor(max_workers=args.transform_workers) as pool:
        futures: dict[Future, tuple[int, str]] = {}
        for case in cases:
            instance_id = case["instance_id"]
            index = index_by_id[instance_id]
            futures[
                pool.submit(
                    prepare_case,
                    index=index,
                    case=case,
                    source_row=source_rows[instance_id],
                    data_dir=args.data_dir,
                    cache_root=args.repo_cache,
                    pyminify=args.pyminify,
                    work_root=work_root,
                    detector=detector,
                    tokenizer=tokenizer,
                    detector_args=detector_args,
                )
            ] = (index, instance_id)
        for future in as_completed(futures):
            index, instance_id = futures[future]
            try:
                rows[instance_id] = future.result()
            except Exception as exc:
                LOGGER.error(
                    "Uncaught transform error for %s: %s\n%s",
                    instance_id,
                    exc,
                    traceback.format_exc(),
                )
                raise
            atomic_write_jsonl(
                args.data_dir / "case_results.jsonl",
                (rows[value] for value in selected_ids if value in rows),
            )

    post_report: dict[str, Any] | None = None
    if not args.prepare_only:
        source_statuses = load_evaluation_statuses(args.source_dir)
        existing_statuses = load_evaluation_statuses(args.data_dir)
        eval_futures: dict[Future, str] = {}
        with ThreadPoolExecutor(max_workers=args.eval_workers) as eval_pool:
            for instance_id in selected_ids:
                row = rows[instance_id]
                index = index_by_id[instance_id]
                existing = existing_statuses.get(instance_id)
                if existing and existing.get("evaluation_complete"):
                    continue
                source_status = source_statuses.get(instance_id)
                if (
                    row["patch_identical"]
                    and not args.no_reuse_identical_evaluations
                    and source_status
                    and source_status.get("evaluation_complete")
                ):
                    copy_reused_evaluation(
                        source_status=source_status,
                        data_dir=args.data_dir,
                        index=index,
                        run_id=args.run_id,
                        row=row,
                    )
                    continue
                case_dir, evaluation_run_id = prepare_async_evaluation(
                    data_dir=args.data_dir,
                    base_run_id=args.run_id,
                    generation_index=index,
                    instance_id=instance_id,
                    row=row,
                )
                eval_futures[
                    eval_pool.submit(
                        evaluate_one_modal,
                        swebench_python=args.swebench_python,
                        case_dir=case_dir,
                        instance_id=instance_id,
                        evaluation_run_id=evaluation_run_id,
                        timeout=args.modal_eval_timeout,
                    )
                ] = instance_id
            for future in as_completed(eval_futures):
                instance_id = eval_futures[future]
                try:
                    future.result()
                except Exception:
                    LOGGER.exception("Evaluator failed for %s", instance_id)
        post_report = aggregate_official_reports(
            data_dir=args.data_dir,
            run_id=args.run_id,
            selected_ids=selected_ids,
            prediction_name=next(iter(rows.values()))["model_name_or_path"],
        )

    summary = make_summary(
        rows=rows,
        selected_ids=selected_ids,
        original_report=original_report,
        post_report=post_report,
        settings={**source_settings, "detection_skipped": args.skip_detection},
    )
    atomic_write_json(args.data_dir / "summary.json", summary)
    LOGGER.info("Python-Minifier summary: %s", json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
