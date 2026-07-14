"""Generate one frozen 100-step trajectory per selected watermarked program."""

from __future__ import annotations

import ast
import io
from pathlib import Path
import token
import tokenize
import traceback
from typing import Any, Mapping

from rw_obfuscator.corpus import execute_case
from rw_obfuscator.engine import RandomWalkObfuscator
from rw_obfuscator.rules import default_rules

from .common import (
    atomic_jsonl,
    config_map,
    index_by_task,
    iter_jsonl,
    load_manifest,
    load_transforms_for_config,
    resolve_repo_path,
    run_root,
    stable_seed,
)


ADVANCED_RULE_FAMILIES = frozenset({"opaque_predicate", "dispatcher"})


def rules_for_profile(rule_profile: str) -> tuple[object, ...]:
    rules = tuple(default_rules())
    if rule_profile == "full":
        return rules
    if rule_profile == "no_advanced":
        return tuple(
            rule
            for rule in rules
            if getattr(rule, "name", None) not in ADVANCED_RULE_FAMILIES
        )
    raise ValueError(f"unknown rule profile: {rule_profile!r}")


def _json_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if value is None:
        return ""
    return str(value)


def _execution_dict(execution: Any) -> dict[str, Any]:
    value = execution.to_dict()
    value["stdout"] = _json_text(value.get("stdout"))
    value["stderr"] = _json_text(value.get("stderr"))
    return value


def _offset(lines: list[str], line: int, column: int) -> int:
    return sum(len(piece) for piece in lines[: line - 1]) + column


def transformed_generation(record: Mapping[str, Any], source: str) -> str:
    """Project a complete transformed program into the paper's detector suffix.

    Instruction-model records score the complete generated script.  Base-model
    records keep their original prompt and score the transformed continuation.
    The old pipeline found that continuation with string splitting; here the
    entry function is located syntactically and the suite colon token defines
    the boundary.
    """

    if bool(record.get("is_inst")):
        return source
    entry_point = record.get("entry_point")
    if not isinstance(entry_point, str) or not entry_point:
        raise ValueError("base-model record is missing entry_point")
    tree = ast.parse(source)
    functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == entry_point
    ]
    if len(functions) != 1:
        raise ValueError(
            f"expected one top-level function {entry_point!r}, found {len(functions)}"
        )
    function_line = functions[0].lineno
    tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    def_index: int | None = None
    for index, item in enumerate(tokens):
        if item.start[0] < function_line:
            continue
        if item.type == token.NAME and item.string == "def":
            following = next(
                (
                    candidate
                    for candidate in tokens[index + 1 :]
                    if candidate.type not in {tokenize.NL, tokenize.COMMENT}
                ),
                None,
            )
            if following is not None and following.type == token.NAME and following.string == entry_point:
                def_index = index
                break
    if def_index is None:
        raise ValueError(f"cannot locate tokenized definition for {entry_point!r}")
    depth = 0
    colon = None
    for item in tokens[def_index + 1 :]:
        if item.type != token.OP:
            continue
        if item.string in "([{":
            depth += 1
        elif item.string in ")]}":
            depth -= 1
        elif item.string == ":" and depth == 0:
            colon = item
            break
    if colon is None:
        raise ValueError(f"cannot locate suite colon for {entry_point!r}")
    lines = source.splitlines(keepends=True)
    tail = source[_offset(lines, colon.end[0], colon.end[1]) :]
    if tail.startswith("\r\n"):
        return tail[2:]
    if tail.startswith("\n"):
        return tail[1:]
    if tail.strip():
        return "    " + tail.lstrip()
    return ""


def transform_record(
    record: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    steps: int,
    global_seed: int,
    timeout: float,
    memory_mb: int,
    rule_profile: str = "full",
    execution_hash_seed: int = 0,
    precomputed_baseline_execution: Any | None = None,
) -> dict[str, Any]:
    source = record.get("solution")
    test = record.get("test")
    record_id = record.get("id")
    if not isinstance(source, str) or not isinstance(test, str) or not isinstance(record_id, str):
        return {
            "schema_version": 1,
            "config_key": config["key"],
            "task_name": record.get("task_name"),
            "record_id": record_id,
            "rule_profile": rule_profile,
            "status": "invalid_input",
            "error": "solution, test, and id must be strings",
        }
    seed = stable_seed(global_seed, record_id)
    programs = [source]
    trace: list[dict[str, Any]] = []
    baseline_execution = precomputed_baseline_execution
    if baseline_execution is None:
        baseline_execution = execute_case(
            source,
            test,
            timeout=timeout,
            memory_mb=memory_mb,
            hash_seed=execution_hash_seed,
        )
    try:
        engine = RandomWalkObfuscator(
            source,
            seed=seed,
            rules=rules_for_profile(rule_profile),
        )
        current = source
        for step_index in range(steps):
            current, step_record = engine.step(current, step_index=step_index)
            programs.append(current)
            trace.append(step_record.to_dict())
        if len(programs) != steps + 1 or len(trace) != steps:
            raise AssertionError("trajectory length mismatch")
        ast.parse(current)
        detector_generation = transformed_generation(record, current)
        execution = execute_case(
            current,
            test,
            timeout=timeout,
            memory_mb=memory_mb,
            hash_seed=execution_hash_seed,
        )
        status = "ok"
        return {
            "schema_version": 1,
            "config_key": config["key"],
            "model_slug": config["model_slug"],
            "watermark": config["watermark"],
            "dataset": config["dataset"],
            "config_id": config["config_id"],
            "task_name": record.get("task_name"),
            "record_id": record_id,
            "seed": seed,
            "steps": steps,
            "rule_profile": rule_profile,
            "status": status,
            "programs": programs,
            "trace": trace,
            "rule_counts": {
                rule: sum(item["rule"] == rule for item in trace)
                for rule in sorted({str(item["rule"]) for item in trace})
            },
            "detection_g4d": detector_generation,
            "baseline": {
                "saved_passed": record.get("passed"),
                "z_score": record.get("z_score"),
                "p_value": record.get("p_value"),
                "execution": _execution_dict(baseline_execution),
            },
            "final": {
                "bytes": len(current.encode("utf-8")),
                "changed": current != source,
                "execution": _execution_dict(execution),
            },
        }
    except Exception as error:
        return {
            "schema_version": 1,
            "config_key": config["key"],
            "model_slug": config["model_slug"],
            "watermark": config["watermark"],
            "dataset": config["dataset"],
            "config_id": config["config_id"],
            "task_name": record.get("task_name"),
            "record_id": record_id,
            "seed": seed,
            "steps": steps,
            "rule_profile": rule_profile,
            "status": "error",
            "programs": programs,
            "trace": trace,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(limit=20),
            "baseline": {
                "saved_passed": record.get("passed"),
                "z_score": record.get("z_score"),
                "p_value": record.get("p_value"),
                "execution": _execution_dict(baseline_execution),
            },
        }


def transform_shard(
    run_id: str,
    shard_index: int,
    *,
    timeout: float = 10.0,
    memory_mb: int = 1024,
    overwrite: bool = False,
) -> Path:
    manifest = load_manifest(run_id)
    shards = manifest["shards"]
    if shard_index < 0 or shard_index >= len(shards):
        raise IndexError(f"shard index {shard_index} outside [0, {len(shards)})")
    shard = shards[shard_index]
    config = config_map(manifest)[shard["config_key"]]
    rule_profile = str(manifest["walk"].get("rule_profile", "full"))
    generate_path = resolve_repo_path(config["generate_path"])
    records = list(iter_jsonl(generate_path))
    selected = records[int(shard["task_start"]) : int(shard["task_stop"])]
    expected = int(shard["task_stop"]) - int(shard["task_start"])
    if len(selected) != expected:
        raise ValueError(f"shard expected {expected} records, got {len(selected)}")
    output = run_root(run_id) / "transforms" / f"part-{shard_index:04d}.jsonl.gz"
    rows = (
        transform_record(
            record,
            config=config,
            steps=int(manifest["walk"]["steps"]),
            global_seed=int(manifest["walk"]["global_seed"]),
            timeout=timeout,
            memory_mb=memory_mb,
            rule_profile=rule_profile,
        )
        for record in selected
    )
    atomic_jsonl(output, rows, overwrite=overwrite)
    return output


def refresh_baseline_shard(
    run_id: str,
    shard_index: int,
    *,
    timeout: float = 10.0,
    memory_mb: int = 1024,
) -> Path:
    """Execute original programs and add same-runtime baseline results.

    This updates only baseline execution metadata.  It deliberately reuses the
    frozen programs and traces, so correcting the utility-comparison cohort
    never changes a trajectory or requires watermark scores to be recomputed.
    """

    manifest = load_manifest(run_id)
    shards = manifest["shards"]
    if shard_index < 0 or shard_index >= len(shards):
        raise IndexError(f"shard index {shard_index} outside [0, {len(shards)})")
    shard = shards[shard_index]
    config = config_map(manifest)[shard["config_key"]]
    records = list(iter_jsonl(resolve_repo_path(config["generate_path"])))
    selected = records[int(shard["task_start"]) : int(shard["task_stop"])]
    records_by_id = {str(record["id"]): record for record in selected}
    output = run_root(run_id) / "transforms" / f"part-{shard_index:04d}.jsonl.gz"
    transformed = list(iter_jsonl(output))
    if len(transformed) != len(selected):
        raise ValueError(
            f"shard {shard_index} has {len(transformed)} transforms for "
            f"{len(selected)} generation records"
        )

    def refreshed_rows():
        for row in transformed:
            record_id = str(row["record_id"])
            record = records_by_id[record_id]
            source = record.get("solution")
            test = record.get("test")
            if not isinstance(source, str) or not isinstance(test, str):
                raise ValueError(f"invalid source/test for {record_id}")
            execution = execute_case(
                source,
                test,
                timeout=timeout,
                memory_mb=memory_mb,
            )
            updated = dict(row)
            baseline = dict(updated.get("baseline", {}))
            if "saved_passed" not in baseline:
                baseline["saved_passed"] = baseline.pop(
                    "passed", record.get("passed")
                )
            baseline["execution"] = _execution_dict(execution)
            updated["baseline"] = baseline
            yield updated

    atomic_jsonl(output, refreshed_rows(), overwrite=True)
    return output


__all__ = [
    "ADVANCED_RULE_FAMILIES",
    "load_transforms_for_config",
    "refresh_baseline_shard",
    "rules_for_profile",
    "transform_record",
    "transform_shard",
    "transformed_generation",
]
