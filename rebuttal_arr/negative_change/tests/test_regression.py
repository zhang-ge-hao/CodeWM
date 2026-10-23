from __future__ import annotations

from pathlib import Path
import sys
import unittest


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from scorer import check_saved_score, load_tokenizer  # noqa: E402
from source_data import (  # noqa: E402
    load_dataset_inputs,
    load_generate_records,
    load_obfuscation_records,
)


class DetectorRegressionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tokenizer = load_tokenizer(local_files_only=True)

    def test_saved_clean_scores_are_reproduced(self) -> None:
        for dataset in ("humaneval_py", "mbpp_py"):
            inputs = load_dataset_inputs(dataset)
            config = inputs.wllm_configs[0]
            generated = load_generate_records(config.generate_path)
            for task in inputs.task_names[:2]:
                with self.subTest(dataset=dataset, task=task):
                    check = check_saved_score(generated[task], tokenizer=self.tokenizer)
                    self.assertTrue(check["matches"], check)

    def test_saved_obfuscated_score_is_reproduced(self) -> None:
        inputs = load_dataset_inputs("humaneval_py")
        config = inputs.wllm_configs[0]
        generated = load_generate_records(config.generate_path)
        obfuscated = load_obfuscation_records(config)
        task = next(
            task
            for task in inputs.task_names
            if obfuscated[task]["pyminify"].get("z_score") is not None
        )
        source = generated[task]
        saved = obfuscated[task]["pyminify"]
        record = {
            **source,
            "p4d": saved["p4d"],
            "g4d": saved["g4d"],
            "z_score": saved["z_score"],
        }
        check = check_saved_score(record, tokenizer=self.tokenizer)
        self.assertTrue(check["matches"], check)


if __name__ == "__main__":
    unittest.main()
