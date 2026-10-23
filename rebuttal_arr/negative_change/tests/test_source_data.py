from __future__ import annotations

from pathlib import Path
import sys
import unittest


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from source_data import (  # noqa: E402
    EXPECTED_CONFIG_IDS,
    load_dataset_inputs,
    load_generate_records,
    load_obfuscation_records,
    read_jsonl,
)


class SourceDataTest(unittest.TestCase):
    def test_reference_and_task_coverage(self) -> None:
        expected = {"humaneval_py": 164, "mbpp_py": 378}
        for dataset, count in expected.items():
            with self.subTest(dataset=dataset):
                inputs = load_dataset_inputs(dataset)
                self.assertEqual(len(inputs.references), count)
                self.assertEqual(len(inputs.task_names), count)
                self.assertEqual(inputs.no_wm_run.temperature, 1.0)
                self.assertEqual(
                    tuple(config.config_id for config in inputs.wllm_configs),
                    EXPECTED_CONFIG_IDS,
                )

    def test_prompts_align_with_no_wm(self) -> None:
        for dataset in ("humaneval_py", "mbpp_py"):
            inputs = load_dataset_inputs(dataset)
            no_wm = load_generate_records(inputs.no_wm_run.generate_path)
            for config in inputs.wllm_configs:
                wllm = load_generate_records(config.generate_path)
                self.assertEqual(set(wllm), set(no_wm))
                self.assertTrue(
                    all(wllm[task]["p4d"] == no_wm[task]["p4d"] for task in wllm)
                )

    def test_reference_construction(self) -> None:
        humaneval = load_dataset_inputs("humaneval_py").references["humaneval_py/0"]
        self.assertEqual(
            humaneval.solution,
            humaneval.prompt + humaneval.canonical_solution,
        )
        mbpp = load_dataset_inputs("mbpp_py").references["mbpp_py/2"]
        self.assertEqual(mbpp.solution, mbpp.canonical_solution)

    def test_reconstructed_positive_cohort_matches_saved_metrics(self) -> None:
        for dataset in ("humaneval_py", "mbpp_py"):
            inputs = load_dataset_inputs(dataset)
            for config in inputs.wllm_configs:
                generated = load_generate_records(config.generate_path)
                obfuscated = load_obfuscation_records(config)
                retained = [
                    task
                    for task in inputs.task_names
                    if generated[task].get("z_score") is not None
                    and obfuscated[task].get("pyminify", {}).get("z_score") is not None
                    and obfuscated[task].get("pyminifier", {}).get("z_score") is not None
                ]
                saved_metrics = read_jsonl(config.directory / "metrics.jsonl")
                with self.subTest(dataset=dataset, config=config.config_id):
                    self.assertEqual(
                        {row["comp_c"] for row in saved_metrics},
                        {len(retained)},
                    )


if __name__ == "__main__":
    unittest.main()
