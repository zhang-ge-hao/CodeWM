from __future__ import annotations

import hashlib
import unittest

import torch

from vllm_wllm import VLLMWatermarkLogitsProcessor, WLLMRequestLogitsProcessor


def reference_wllm(
    prompt: list[int],
    output: list[int],
    scores: torch.Tensor,
    *,
    gamma: float,
    delta: float,
    ngram_len: int,
    hash_key: int,
    vocab_size: int | None = None,
) -> torch.Tensor:
    prefix = prompt + output
    previous = [prefix[-index] for index in range(1, ngram_len)]
    encoded = ",".join(map(str, previous)).encode()
    seed = int(hashlib.sha3_512(encoded).hexdigest(), 16) & 0xFFFFFFFFFFFFFFFF
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed ^ hash_key)
    effective_vocab_size = vocab_size or scores.shape[-1]
    green = torch.randperm(effective_vocab_size, generator=rng)[
        : int(effective_vocab_size * gamma)
    ]
    expected = scores.clone()
    expected[green] += delta
    return expected


class WLLMvLLMParityTest(unittest.TestCase):
    def test_matches_existing_seed_and_greenlist_rule(self) -> None:
        prompt = [101, 202, 303, 404, 505]
        for output in ([], [606], [606, 707, 808]):
            for gamma, delta in ((0.1, 0.5), (0.25, 2.0), (0.5, 4.0)):
                with self.subTest(output=output, gamma=gamma, delta=delta):
                    scores = torch.linspace(-3.0, 3.0, 257)
                    processor = WLLMRequestLogitsProcessor(
                        gamma=gamma,
                        delta=delta,
                        ngram_len=5,
                        hash_key=15485863,
                    )
                    actual = processor(prompt, output, scores.clone())
                    expected = reference_wllm(
                        prompt,
                        output,
                        scores,
                        gamma=gamma,
                        delta=delta,
                        ngram_len=5,
                        hash_key=15485863,
                    )
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_short_prefix_is_unchanged(self) -> None:
        scores = torch.randn(100)
        processor = WLLMRequestLogitsProcessor(
            gamma=0.5,
            delta=4.0,
            ngram_len=5,
            hash_key=15485863,
        )
        torch.testing.assert_close(
            processor([1, 2], [3], scores.clone()), scores, rtol=0, atol=0
        )

    def test_uses_tokenizer_vocab_not_padded_logits_width(self) -> None:
        scores = torch.zeros(263)
        processor = WLLMRequestLogitsProcessor(
            gamma=0.5,
            delta=4.0,
            ngram_len=5,
            hash_key=15485863,
            vocab_size=257,
        )
        actual = processor([101, 202, 303, 404], [], scores.clone())
        expected = reference_wllm(
            [101, 202, 303, 404],
            [],
            scores,
            gamma=0.5,
            delta=4.0,
            ngram_len=5,
            hash_key=15485863,
            vocab_size=257,
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(actual[257:].count_nonzero().item(), 0)

    def test_openai_xargs_boolean_normalization(self) -> None:
        self.assertTrue(VLLMWatermarkLogitsProcessor._enabled(True))
        self.assertTrue(VLLMWatermarkLogitsProcessor._enabled("true"))
        self.assertTrue(VLLMWatermarkLogitsProcessor._enabled(1))
        self.assertFalse(VLLMWatermarkLogitsProcessor._enabled(False))
        self.assertFalse(VLLMWatermarkLogitsProcessor._enabled("false"))
        self.assertFalse(VLLMWatermarkLogitsProcessor._enabled(0))
        with self.assertRaises(ValueError):
            VLLMWatermarkLogitsProcessor._enabled("not-a-boolean")


if __name__ == "__main__":
    unittest.main()
