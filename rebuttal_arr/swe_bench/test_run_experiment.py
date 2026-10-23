from __future__ import annotations

import sys
from pathlib import Path
import unittest

import torch


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_experiment as experiment  # noqa: E402


class TinyTokenizer:
    def __init__(self, vocab_size: int) -> None:
        self._vocab = {f"token-{index}": index for index in range(vocab_size)}

    def get_vocab(self) -> dict[str, int]:
        return self._vocab


def settings(watermarking: str) -> experiment.GenerationSettings:
    return experiment.GenerationSettings(
        model_id="test-model",
        watermarking=watermarking,
        prompt_transport="raw",
        max_input_tokens=100,
        max_new_tokens=100,
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        gamma=0.5 if watermarking == "wllm" else None,
        delta=4.0 if watermarking == "wllm" else None,
        ngram_len=experiment.NGRAM_LEN,
        watermark_key=15485863,
        z_threshold=4.0,
        generation_seed=1,
    )


class WatermarkFlowTest(unittest.TestCase):
    def test_non_wm_has_no_processor_or_detector(self) -> None:
        logits, detector, vocab_size = experiment.create_watermark_components(
            TinyTokenizer(100), settings("none")
        )
        self.assertIsNone(logits)
        self.assertIsNone(detector)
        self.assertEqual(vocab_size, 100)
        self.assertTrue(experiment.prediction_model_name(settings("none")).endswith("non-wm"))

    def test_detector_exactly_matches_processor_token_history(self) -> None:
        config = settings("wllm")
        logits, detector, _ = experiment.create_watermark_components(
            TinyTokenizer(100), config
        )
        self.assertIsNotNone(logits)
        self.assertIsNotNone(detector)

        prefix = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.long)
        sequence = prefix.clone()
        generated: list[int] = []
        for _ in range(100):
            scores = torch.zeros((1, 100), dtype=torch.float32)
            biased = logits(sequence.unsqueeze(0), scores)
            token = int(torch.argmax(biased[0]).item())
            generated.append(token)
            sequence = torch.cat((sequence, torch.tensor([token])))

        result = experiment.detect_completion(
            detector=detector,
            prompt_ids=prefix,
            generated_ids=generated,
            settings=config,
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["num_tokens_scored"], 100)
        self.assertEqual(result["num_green_tokens"], 100)
        self.assertAlmostEqual(result["green_fraction"], 1.0)
        self.assertGreater(result["z_score"], 4.0)
        self.assertTrue(result["prediction"])
        self.assertGreater(experiment.ideal_auroc([result["z_score"]]), 0.99)

    def test_ideal_auroc_against_standard_normal(self) -> None:
        self.assertAlmostEqual(experiment.ideal_auroc([0.0]), 0.5)
        self.assertGreater(experiment.ideal_auroc([2.0, 3.0]), 0.98)


if __name__ == "__main__":
    unittest.main()
