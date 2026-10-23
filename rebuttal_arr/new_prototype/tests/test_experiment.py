from __future__ import annotations

from collections import Counter

import pytest

from experiment import transform as transform_module
from experiment.common import build_manifest, discover_selected_configs, stable_seed
from experiment.transform import (
    ADVANCED_RULE_FAMILIES,
    _json_text,
    rules_for_profile,
    transform_record,
    transformed_generation,
)
from rw_obfuscator import RandomWalkObfuscator
from rw_obfuscator.corpus import ExecutionResult


ADVANCED_ACTIONS = {
    "insert_true_opaque_guard",
    "remove_true_opaque_guard",
    "insert_false_opaque_guard",
    "remove_false_opaque_guard",
    "flatten_straight_line",
    "restore_straight_line",
    "flatten_simple_if",
    "restore_simple_if",
}


def test_filtered_manifest_has_frozen_counts() -> None:
    manifest = build_manifest(run_id="unit-test")
    assert manifest["counts"] == {
        "configs": 43,
        "configs_by_scheme": {"wllm": 13, "sweet": 24, "synthid": 6},
        "walks": 11118,
        "walks_by_scheme": {"wllm": 3416, "sweet": 6076, "synthid": 1626},
        "transitions": 1111800,
        "saved_programs": 1122918,
        "transform_shards": 124,
    }
    assert all(shard["task_stop"] - shard["task_start"] <= 100 for shard in manifest["shards"])
    assert manifest["walk"]["rule_profile"] == "full"


def test_no_advanced_manifest_records_profile_without_changing_counts() -> None:
    full = build_manifest(run_id="unit-full", tasks_per_shard=25)
    ablation = build_manifest(
        run_id="unit-no-advanced",
        tasks_per_shard=25,
        rule_profile="no_advanced",
    )
    assert ablation["walk"]["rule_profile"] == "no_advanced"
    assert ablation["counts"] == full["counts"]
    assert [item["key"] for item in ablation["configs"]] == [
        item["key"] for item in full["configs"]
    ]
    assert ablation["input_files"] == full["input_files"]


def test_no_advanced_profile_removes_both_complete_rule_families() -> None:
    full_names = {getattr(rule, "name", None) for rule in rules_for_profile("full")}
    ablated_names = {
        getattr(rule, "name", None) for rule in rules_for_profile("no_advanced")
    }
    assert ADVANCED_RULE_FAMILIES <= full_names
    assert ADVANCED_RULE_FAMILIES.isdisjoint(ablated_names)
    assert ablated_names == full_names - ADVANCED_RULE_FAMILIES


def test_no_advanced_profile_enumerates_no_advanced_actions() -> None:
    source = "def f(x):\n    y=x+1\n    z=y*2\n    return z\n"
    full = RandomWalkObfuscator(
        source,
        seed=3,
        rules=rules_for_profile("full"),
    )
    ablated = RandomWalkObfuscator(
        source,
        seed=3,
        rules=rules_for_profile("no_advanced"),
    )
    full_actions = {action.rule for action in full.enumerate_actions(source)}
    ablated_actions = {action.rule for action in ablated.enumerate_actions(source)}
    assert full_actions & ADVANCED_ACTIONS
    assert ADVANCED_ACTIONS.isdisjoint(ablated_actions)


def test_filtered_config_cells_match_expected_table() -> None:
    counts = Counter(
        (config.watermark, config.model_slug, config.dataset)
        for config in discover_selected_configs()
    )
    assert counts == Counter(
        {
            ("wllm", "Llama31Instruct8B", "humaneval_py"): 7,
            ("wllm", "Llama31Instruct8B", "mbpp_py"): 6,
            ("sweet", "DSCoderBase33B", "humaneval_py"): 1,
            ("sweet", "Llama31Instruct8B", "humaneval_py"): 13,
            ("sweet", "Llama31Instruct8B", "mbpp_py"): 10,
            ("synthid", "DSCoderBase33B", "humaneval_py"): 2,
            ("synthid", "DSCoderBase33B", "mbpp_py"): 1,
            ("synthid", "Llama31Instruct8B", "humaneval_py"): 1,
            ("synthid", "Llama31Instruct8B", "mbpp_py"): 2,
        }
    )


def test_stable_seed_is_deterministic_and_record_specific() -> None:
    assert stable_seed(10771, "record-a") == stable_seed(10771, "record-a")
    assert stable_seed(10771, "record-a") != stable_seed(10771, "record-b")
    assert stable_seed(10771, "record-a") != stable_seed(10772, "record-a")


def test_timeout_stream_bytes_are_json_safe() -> None:
    assert _json_text(b"valid\xff") == "valid\ufffd"


def test_transformed_generation_instruction_uses_complete_program() -> None:
    source = "def answer():\n    return 42\n"
    assert transformed_generation({"is_inst": True}, source) == source


def test_transformed_generation_base_extracts_multiline_suite_and_tail() -> None:
    source = (
        "from typing import List\n\n"
        "def answer(values: List[int]) -> int:\n"
        "    total = sum(values)\n"
        "    return total\n\n"
        "SENTINEL = 1\n"
    )
    generation = transformed_generation(
        {"is_inst": False, "entry_point": "answer"}, source
    )
    assert generation.startswith("    total = sum(values)\n")
    assert generation.endswith("SENTINEL = 1\n")


def test_transformed_generation_base_expands_inline_suite_for_detector() -> None:
    source = "def answer(value): return value + 1\n"
    assert transformed_generation(
        {"is_inst": False, "entry_point": "answer"}, source
    ) == "    return value + 1\n"


def test_transform_record_saves_every_state_and_only_final_execution() -> None:
    source = "def answer(value):\n    result = value + 1\n    return result\n"
    record = {
        "id": "unit-record",
        "task_name": "unit/0",
        "solution": source,
        "test": "assert answer(2) == 3\n",
        "entry_point": "answer",
        "is_inst": True,
        "passed": True,
        "z_score": 1.0,
        "p_value": 0.1,
    }
    config = {
        "key": "unit-config",
        "model_slug": "unit-model",
        "watermark": "wllm",
        "dataset": "unit",
        "config_id": "001",
    }
    result = transform_record(
        record,
        config=config,
        steps=3,
        global_seed=10771,
        timeout=2.0,
        memory_mb=512,
    )
    assert result["status"] == "ok"
    assert len(result["programs"]) == 4
    assert len(result["trace"]) == 3
    assert result["baseline"]["execution"]["passed"] is True
    assert result["final"]["execution"]["passed"] is True
    assert all(0 <= len(program.encode("utf-8")) < 2000 for program in result["programs"])


def test_transform_record_reuses_precomputed_baseline(monkeypatch) -> None:
    execution = ExecutionResult(
        passed=True,
        returncode=0,
        timed_out=False,
        duration_seconds=0.01,
        stdout="",
        stderr="",
    )
    calls = []

    def fake_execute_case(*args, **kwargs):
        calls.append((args, kwargs))
        return execution

    monkeypatch.setattr(transform_module, "execute_case", fake_execute_case)
    result = transform_record(
        {
            "id": "unit-cached-baseline",
            "task_name": "unit/cached",
            "solution": "def answer(value):\n    return value + 1\n",
            "test": "assert answer(2) == 3\n",
            "entry_point": "answer",
            "is_inst": True,
            "passed": True,
            "z_score": 1.0,
            "p_value": 0.1,
        },
        config={
            "key": "unit-config",
            "model_slug": "unit-model",
            "watermark": "wllm",
            "dataset": "unit",
            "config_id": "001",
        },
        steps=0,
        global_seed=10771,
        timeout=2.0,
        memory_mb=512,
        precomputed_baseline_execution=execution,
    )
    assert result["status"] == "ok"
    assert len(calls) == 1
    assert result["baseline"]["execution"]["passed"] is True


def test_transform_record_no_advanced_trace_contains_no_advanced_actions() -> None:
    source = "def answer(value):\n    x=value+1\n    y=x*2\n    return y\n"
    record = {
        "id": "unit-no-advanced-record",
        "task_name": "unit/1",
        "solution": source,
        "test": "assert answer(2) == 6\n",
        "entry_point": "answer",
        "is_inst": True,
        "passed": True,
        "z_score": 1.0,
        "p_value": 0.1,
    }
    config = {
        "key": "unit-config",
        "model_slug": "unit-model",
        "watermark": "wllm",
        "dataset": "unit",
        "config_id": "001",
    }
    result = transform_record(
        record,
        config=config,
        steps=20,
        global_seed=10771,
        timeout=2.0,
        memory_mb=512,
        rule_profile="no_advanced",
    )
    assert result["status"] == "ok"
    assert result["rule_profile"] == "no_advanced"
    assert ADVANCED_ACTIONS.isdisjoint(item["rule"] for item in result["trace"])
