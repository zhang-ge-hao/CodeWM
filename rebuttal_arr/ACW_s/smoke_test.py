#!/usr/bin/env python3
"""Small equivalence checks against the vendored official implementation."""

from __future__ import annotations

from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "upstream"))

from acw_s import ACWSDetector, ACWSLogitsProcessor  # noqa: E402
from run_upstream import empty_detection_result  # noqa: E402
from models.sweetcode import (  # noqa: E402
    SweetCodeDetector,
    SweetCodeLogitsProcessor,
)


def assert_close(left, right, tolerance=1e-6):
    if abs(float(left) - float(right)) > tolerance:
        raise AssertionError(f"{left!r} != {right!r}")


def check_processor(vocab_size: int) -> None:
    input_ids = torch.tensor([[2, 7, 3], [4, 8, 5]], dtype=torch.long)
    generator = torch.Generator().manual_seed(1234 + vocab_size)
    scores = torch.randn((2, vocab_size), generator=generator)

    official = SweetCodeLogitsProcessor(
        vocab_size=vocab_size,
        gamma=0.5,
        delta=2.0,
        entropy_threshold=0.0,
    )
    audited = ACWSLogitsProcessor(
        vocab_size=vocab_size,
        gamma=0.5,
        delta=2.0,
        entropy_threshold=0.0,
    )
    official_output = official(input_ids.clone(), scores.clone())
    audited_output = audited(input_ids.clone(), scores.clone())
    if not torch.equal(official_output, audited_output):
        raise AssertionError(f"processor mismatch for vocabulary {vocab_size}")


def check_detector() -> None:
    vocab_size = 12
    token_ids = torch.tensor([1, 2, 4, 3, 8, 7, 5, 11], dtype=torch.long)
    generator = torch.Generator().manual_seed(2026)
    source_scores = torch.randn((len(token_ids), vocab_size), generator=generator)
    entropies = [0.0, 0.0, 2.0, 0.1, 1.7, 2.3, 0.2, 1.8]
    prefix_len = 2

    official = SweetCodeDetector(
        vocab_size=vocab_size,
        gamma=0.5,
        delta=2.0,
        entropy_threshold=1.2,
        z_threshold=4.0,
        tokenizer=None,
    )
    audited = ACWSDetector(
        vocab_size=vocab_size,
        gamma=0.5,
        delta=2.0,
        entropy_threshold=1.2,
        z_threshold=4.0,
    )
    official_result = official.detect(
        tokenized_text=token_ids,
        prefix_len=prefix_len,
        entropy=entropies,
        scores=source_scores,
    )
    audited_result = audited.detect(
        token_ids=token_ids,
        prefix_len=prefix_len,
        entropies=entropies,
        source_scores=source_scores,
    )
    for key in (
        "num_tokens_generated",
        "num_tokens_scored",
        "num_green_tokens",
        "watermarking_fraction",
        "green_fraction",
        "z_score",
        "p_value",
        "prediction",
    ):
        if isinstance(official_result[key], bool):
            if official_result[key] != audited_result[key]:
                raise AssertionError(f"detector mismatch for {key}")
        else:
            assert_close(official_result[key], audited_result[key])


def check_empty_result() -> None:
    result = empty_detection_result(
        {"pass@1": 0.0, "pass@10": 0.0},
        {"0": []},
    )
    assert result["empty_generation"] is True
    assert result["prediction"] is False
    assert result["z_score"] == 0.0
    assert result["num_tokens_generated"] == 0
    assert result["pass@1"] == 0.0


def main() -> None:
    check_processor(12)
    check_processor(13)
    check_detector()
    check_empty_result()
    print("ACW-s audited implementation matches the vendored official code")


if __name__ == "__main__":
    main()
