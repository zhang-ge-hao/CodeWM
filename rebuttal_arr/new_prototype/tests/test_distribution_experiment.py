from __future__ import annotations

import numpy as np
import json

from distribution_experiment import all_cases, run as distribution_run
from distribution_experiment.run import (
    build_manifest,
    build_rejected_followup_manifest,
    build_useful_parseable_sample_manifest,
    discover_candidates,
    fixed_standard_normal_ad,
)
from distribution_experiment.all_cases import build_manifest as build_all_cases_manifest
from distribution_experiment.all_cases import write_sorted_results


def test_candidate_universe_and_sample_are_frozen() -> None:
    candidates, _ = discover_candidates()
    assert len(candidates) == 1620
    manifest = build_manifest("unit-distribution")
    assert manifest["counts"]["sampled_seeds"] == 100
    assert manifest["counts"]["trajectories"] == 3000
    assert manifest["counts"]["transitions"] == 300000
    assert sum(manifest["counts"]["sampled_seeds_by_scheme"].values()) == 100
    assert len({row["candidate_key"] for row in manifest["candidates"]}) == 100
    assert "fixed_standard_normal_monte_carlo_draws" not in manifest["test"]


def test_fixed_standard_normal_ad_distinguishes_large_shift() -> None:
    rng = np.random.default_rng(10771)
    standard = rng.standard_normal(30)
    shifted = standard + 5.0
    assert fixed_standard_normal_ad(standard) < fixed_standard_normal_ad(shifted)


def test_rejected_followup_manifest_is_frozen_to_the_seven_paper_ad_rejections() -> None:
    manifest = build_rejected_followup_manifest(
        "unit-rw500-followup",
        "rw100-z4-sample100-v2",
        steps=500,
    )
    assert manifest["selection"]["source_hypothesis_decision"] == "reject normality null"
    assert manifest["walk"]["steps"] == 500
    assert manifest["walk"]["trajectories_per_seed"] == 30
    assert manifest["counts"]["sampled_seeds"] == 7
    assert manifest["counts"]["sampled_seeds_by_scheme"] == {"sweet": 5, "wllm": 2}
    assert manifest["counts"]["trajectories"] == 210
    assert manifest["counts"]["transitions"] == 105_000
    assert manifest["counts"]["transform_shards"] == 210
    assert "fixed_standard_normal_monte_carlo_draws" not in manifest["test"]
    assert manifest["selection"]["verify_source_prefix"] is True


def test_rejected_repeat_can_use_independent_walk_seed() -> None:
    manifest = build_rejected_followup_manifest(
        "unit-rw100-repeat",
        "rw100-z4-sample100-v2",
        steps=100,
        global_seed=10_772,
        verify_source_prefix=False,
    )
    assert manifest["counts"]["sampled_seeds"] == 7
    assert manifest["counts"]["trajectories"] == 210
    assert manifest["counts"]["transitions"] == 21_000
    assert manifest["walk"]["global_seed"] == 10_772
    assert manifest["selection"]["verify_source_prefix"] is False


def test_all_space_followup_manifest_extends_the_frozen_sample() -> None:
    manifest = build_rejected_followup_manifest(
        "unit-rw500-all",
        "rw100-z4-sample100-v2",
        steps=500,
        selection="all",
    )
    assert manifest["selection"]["selection_mode"] == "all"
    assert manifest["counts"]["sampled_seeds"] == 100
    assert manifest["counts"]["sampled_seeds_by_scheme"] == {
        "sweet": 51,
        "synthid": 4,
        "wllm": 45,
    }
    assert manifest["counts"]["trajectories"] == 3_000
    assert manifest["counts"]["transitions"] == 1_500_000
    assert manifest["counts"]["transform_shards"] == 3_000


def test_useful_sample_keeps_failed_but_parseable_programs() -> None:
    manifest = build_useful_parseable_sample_manifest("unit-useful-sample")
    assert manifest["counts"]["source_records"] == 11_118
    assert manifest["counts"]["parseable_candidates"] == 10_877
    assert manifest["counts"]["unparseable_excluded"] == 241
    assert manifest["counts"]["sampled_seeds"] == 100
    assert sum(manifest["counts"]["sampled_seeds_by_original_test"].values()) == 100
    assert manifest["counts"]["sampled_seeds_by_original_test"]["failed"] > 0
    assert manifest["counts"]["trajectories"] == 3_000
    assert manifest["counts"]["transitions"] == 300_000
    assert manifest["counts"]["transform_shards"] == 375
    assert manifest["test"]["require_baseline_passed"] is False
    assert manifest["test"]["require_same_test_outcome"] is True


def test_all_case_manifest_has_random_seed_per_case_and_no_timestamp() -> None:
    manifest = build_all_cases_manifest("unit-all-cases")
    assert manifest["counts"]["cases"] == 1_620
    assert manifest["counts"]["cases_by_scheme"] == {
        "sweet": 867,
        "synthid": 53,
        "wllm": 700,
    }
    assert manifest["counts"]["configs"] == 186
    assert manifest["counts"]["trajectories"] == 48_600
    assert manifest["counts"]["transitions"] == 4_860_000
    assert manifest["counts"]["transform_shards"] == 4_860
    assert all(
        shard["task_stop"] - shard["task_start"] == 10
        for shard in manifest["shards"]
    )
    seeds = [candidate["random_seed"] for candidate in manifest["candidates"]]
    assert len(seeds) == len(set(seeds)) == 1_620
    assert all(isinstance(seed, int) and seed > 0 for seed in seeds)
    assert "created_utc" not in manifest
    assert manifest["test"]["require_baseline_passed"] is False
    assert manifest["test"]["require_same_test_outcome"] is True


def test_result_writer_sorts_and_atomically_overwrites(tmp_path) -> None:
    path = tmp_path / "results.jsonl"
    write_sorted_results(
        path,
        [
            {"key": "z", "random_seed": 2},
            {"key": "a", "random_seed": 1},
        ],
    )
    assert [json.loads(line)["key"] for line in path.read_text().splitlines()] == ["a", "z"]
    write_sorted_results(path, [{"key": "m", "random_seed": 3}])
    assert [json.loads(line)["key"] for line in path.read_text().splitlines()] == ["m"]


def test_result_row_keeps_programs_aligned_with_endpoint_scores() -> None:
    candidate = {
        "candidate_key": "config::task",
        "random_seed": 123,
        "record_id": "record",
        "config_key": "config",
        "task_name": "task",
        "watermark": "sweet",
        "model_slug": "model",
        "dataset": "humaneval_py",
        "original_z_score": 5.0,
    }
    scores = [
        {
            "trajectory_index": index,
            "score": {"z_score": index / 10.0},
            "saved_original_passed": True,
            "baseline_execution_passed": True,
            "final_execution_passed": True,
        }
        for index in reversed(range(30))
    ]
    programs = {index: f"# endpoint {index}" for index in range(30)}

    result = all_cases._result_row(candidate, scores, programs)

    assert result["endpoint_z_scores"] == [index / 10.0 for index in range(30)]
    assert result["endpoint_programs"] == [
        f"# endpoint {index}" for index in range(30)
    ]


def test_score_assignment_balances_by_trajectory_count() -> None:
    manifest = {
        "walk": {"trajectories_per_seed": 30},
        "configs": [
            {"key": key, "watermark": "wllm", "model_slug": "model"}
            for key in ("a", "b", "c", "d")
        ],
        "candidates": [
            *({"config_key": "a"} for _ in range(10)),
            *({"config_key": "b"} for _ in range(9)),
            {"config_key": "c"},
            {"config_key": "d"},
        ],
    }
    assignments = [
        distribution_run.assigned_config_keys(
            manifest,
            scheme="wllm",
            model_slug=None,
            shard_index=index,
            shard_count=2,
        )
        for index in range(2)
    ]
    assert assignments == [["a", "d"], ["b", "c"]]
    assert {key for assignment in assignments for key in assignment} == {
        "a",
        "b",
        "c",
        "d",
    }


def test_validation_endpoint_index_keeps_only_scoring_fields(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(distribution_run, "run_root", lambda _run_id: tmp_path)
    manifest = {
        "walk": {"trajectories_per_seed": 2},
        "candidates": [
            {"config_key": "config-a"},
            {"config_key": "config-b"},
        ],
    }
    rows = []
    for config_key in ("config-a", "config-b"):
        for trajectory_index in range(2):
            rows.append(
                {
                    "config_key": config_key,
                    "candidate_key": f"{config_key}::task",
                    "trajectory_index": trajectory_index,
                    "trajectory_id": f"{config_key}::task::{trajectory_index}",
                    "status": "ok",
                    "programs": ["original", f"endpoint-{trajectory_index}"],
                    "detection_g4d": f"detector-{trajectory_index}",
                    "baseline": {
                        "saved_passed": True,
                        "execution": {"passed": True},
                    },
                    "final": {"execution": {"passed": True}},
                }
            )

    assert distribution_run._write_endpoint_index("unit", manifest, rows) == 2
    indexed = list(
        distribution_run.iter_jsonl(
            tmp_path / "endpoints" / "config-a.jsonl.gz"
        )
    )
    assert [row["endpoint_program"] for row in indexed] == [
        "endpoint-0",
        "endpoint-1",
    ]
    assert all("programs" not in row and "trace" not in row for row in indexed)
