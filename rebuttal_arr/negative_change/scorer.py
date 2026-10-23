"""Read-only WLLM scoring for the empirical-negative experiment.

This module deliberately imports only the repository's WLLM detector, not the
stateful experiment pipeline in ``src/detection.py`` or ``src/_hf_obj.py``.
It nevertheless preserves that pipeline's tokenization and invalid-result
semantics so newly scored negatives are directly comparable with saved WLLM
positive scores.
"""

from __future__ import annotations

from functools import lru_cache
import math
from pathlib import Path
import sys
from typing import Any, Mapping

import scipy.stats
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Importing _sweet is intentional: it is the detector implementation used by
# the original experiment.  Do not replace this with src.detection, whose
# detect() skips no-WM samples and mutates task objects.
from _sweet import WatermarkDetector  # noqa: E402


DEFAULT_TOKENIZER = "meta-llama/Llama-3.1-8B-Instruct"


@lru_cache(maxsize=None)
def _load_tokenizer_cached(
    tokenizer_name_or_path: str,
    local_files_only: bool,
) -> PreTrainedTokenizerBase:
    """Load and cache the tokenizer without loading model weights."""

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        local_files_only=local_files_only,
    )
    # This is the only post-load modification made by get_hf_tokenizer() in
    # the original pipeline and can affect padding behavior.
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_tokenizer(
    tokenizer_name_or_path: str | Path = DEFAULT_TOKENIZER,
    *,
    local_files_only: bool = True,
) -> PreTrainedTokenizerBase:
    """Return the LLaMA tokenizer, using the local HF cache by default.

    Set ``local_files_only=False`` only for the one-time, explicitly authorized
    cache initialization.  Normal experiment runs should remain local-only.
    """

    return _load_tokenizer_cached(
        str(tokenizer_name_or_path),
        bool(local_files_only),
    )


def tokenize_segment(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
) -> torch.Tensor:
    """Tokenize one detection segment exactly as ``src/detection.py`` does."""

    if not isinstance(text, str):
        raise TypeError(f"Detection text must be str, got {type(text).__name__}")
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    return inputs["input_ids"].squeeze()


def tokenize_for_detection(
    p4d: str,
    g4d: str,
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(tokenized_text, tokenized_prefix)`` for WLLM detection.

    The prompt and generation are intentionally tokenized separately before
    concatenation.  Tokenizing ``p4d + g4d`` in one call is not equivalent for
    this tokenizer and would not reproduce the paper's saved scores.
    """

    tokenized_prefix = tokenize_segment(p4d, tokenizer)
    tokenized_suffix = tokenize_segment(g4d, tokenizer)

    # Keep the original pipeline's behavior for empty text or a scalar tensor
    # produced by squeeze() from a one-token suffix.
    if (
        len(g4d) == 0
        or len(tokenized_suffix.size()) == 0
        or tokenized_suffix.size(-1) == 0
    ):
        tokenized_text = tokenized_prefix
    else:
        tokenized_text = torch.cat(
            (tokenized_prefix, tokenized_suffix),
            dim=0,
        )
    return tokenized_text, tokenized_prefix


def _format_detection_result(raw: Mapping[str, Any], z_threshold: float) -> dict[str, Any]:
    invalid = bool(raw.get("invalid", False))
    z_score = 0.0 if invalid else float(raw["z_score"])
    p_value = float(scipy.stats.norm.sf(z_score))
    return {
        "z_score": z_score,
        "p_value": p_value,
        "num_tokens_scored": (
            None
            if raw.get("num_tokens_scored") is None
            else int(raw["num_tokens_scored"])
        ),
        "num_green_tokens": (
            None
            if raw.get("num_green_tokens") is None
            else int(raw["num_green_tokens"])
        ),
        "green_fraction": (
            None
            if raw.get("green_fraction") is None
            else float(raw["green_fraction"])
        ),
        "invalid": invalid,
        "prediction": bool((not invalid) and z_score > float(z_threshold)),
    }


class WllmConfigScorer:
    """Reusable scorer for one gamma/n-gram configuration.

    A config scorer caches the 128k-token vocabulary, detector, tokenizer, and
    prompt tokenization.  Only the saved per-task key changes between calls.
    It is intentionally sequential and not thread-safe.
    """

    def __init__(
        self,
        *,
        gamma: float,
        ngram_len: int,
        tokenizer: PreTrainedTokenizerBase | None = None,
        tokenizer_name_or_path: str | Path = DEFAULT_TOKENIZER,
        local_files_only: bool = True,
        z_threshold: float = 4.0,
    ) -> None:
        if not 0.0 < float(gamma) < 1.0:
            raise ValueError(f"gamma must be in (0, 1), got {gamma!r}")
        if int(ngram_len) < 1:
            raise ValueError(f"ngram_len must be positive, got {ngram_len!r}")
        self.tokenizer = tokenizer or load_tokenizer(
            tokenizer_name_or_path,
            local_files_only=local_files_only,
        )
        self.z_threshold = float(z_threshold)
        self.detector = WatermarkDetector(
            vocab=list(self.tokenizer.get_vocab().values()),
            gamma=float(gamma),
            tokenizer=self.tokenizer,
            z_threshold=self.z_threshold,
            ngram_len=int(ngram_len),
            hash_key=0,
        )

    def _score_with_prefix(
        self,
        tokenized_prefix: torch.Tensor,
        g4d: str,
        *,
        custom_seed: int,
    ) -> dict[str, Any]:
        tokenized_suffix = tokenize_segment(g4d, self.tokenizer)
        if (
            len(g4d) == 0
            or len(tokenized_suffix.size()) == 0
            or tokenized_suffix.size(-1) == 0
        ):
            tokenized_text = tokenized_prefix
        else:
            tokenized_text = torch.cat((tokenized_prefix, tokenized_suffix), dim=0)
        self.detector.hash_key = int(custom_seed)
        raw = self.detector.detect(
            tokenized_text=tokenized_text,
            tokenized_prefix=tokenized_prefix,
            return_green_token_mask=False,
        )
        return _format_detection_result(raw, self.z_threshold)

    def score(self, p4d: str, g4d: str, *, custom_seed: int) -> dict[str, Any]:
        prefix = tokenize_segment(p4d, self.tokenizer)
        return self._score_with_prefix(prefix, g4d, custom_seed=custom_seed)

    def score_many(
        self,
        p4d: str,
        generations: Mapping[str, str],
        *,
        custom_seed: int,
    ) -> dict[str, dict[str, Any]]:
        prefix = tokenize_segment(p4d, self.tokenizer)
        return {
            name: self._score_with_prefix(prefix, g4d, custom_seed=custom_seed)
            for name, g4d in generations.items()
        }


def score_wllm(
    p4d: str,
    g4d: str,
    *,
    gamma: float,
    ngram_len: int,
    custom_seed: int,
    tokenizer: PreTrainedTokenizerBase | None = None,
    tokenizer_name_or_path: str | Path = DEFAULT_TOKENIZER,
    local_files_only: bool = True,
    z_threshold: float = 4.0,
) -> dict[str, Any]:
    """Score text with one saved WLLM detector key/configuration.

    The returned mapping is deliberately compact and JSON-serializable.  As in
    ``src/detection.py``, a detector-invalid sequence receives ``z_score=0``
    and the corresponding one-sided normal-tail ``p_value=0.5``.
    """

    config_scorer = WllmConfigScorer(
        gamma=float(gamma),
        tokenizer=tokenizer,
        z_threshold=float(z_threshold),
        ngram_len=int(ngram_len),
        tokenizer_name_or_path=tokenizer_name_or_path,
        local_files_only=local_files_only,
    )
    return config_scorer.score(p4d, g4d, custom_seed=custom_seed)


def score_record(
    record: Mapping[str, Any],
    *,
    tokenizer: PreTrainedTokenizerBase | None = None,
    tokenizer_name_or_path: str | Path = DEFAULT_TOKENIZER,
    local_files_only: bool = True,
    z_threshold: float = 4.0,
) -> dict[str, Any]:
    """Score a result-like mapping; useful for saved-score regression tests."""

    required = ("p4d", "g4d", "gamma", "ngram_len", "custom_seed")
    missing = [key for key in required if record.get(key) is None]
    if missing:
        raise KeyError(f"Record is missing required fields: {', '.join(missing)}")
    return score_wllm(
        record["p4d"],
        record["g4d"],
        gamma=record["gamma"],
        ngram_len=record["ngram_len"],
        custom_seed=record["custom_seed"],
        tokenizer=tokenizer,
        tokenizer_name_or_path=tokenizer_name_or_path,
        local_files_only=local_files_only,
        z_threshold=z_threshold,
    )


def check_saved_score(
    record: Mapping[str, Any],
    *,
    tokenizer: PreTrainedTokenizerBase | None = None,
    tokenizer_name_or_path: str | Path = DEFAULT_TOKENIZER,
    local_files_only: bool = True,
    atol: float = 1e-12,
) -> dict[str, Any]:
    """Recompute a saved result and report whether its z-score is reproduced."""

    if record.get("z_score") is None:
        raise KeyError("Record is missing saved z_score")
    result = score_record(
        record,
        tokenizer=tokenizer,
        tokenizer_name_or_path=tokenizer_name_or_path,
        local_files_only=local_files_only,
    )
    saved = float(record["z_score"])
    recomputed = float(result["z_score"])
    abs_error = abs(saved - recomputed)
    return {
        "matches": math.isclose(saved, recomputed, rel_tol=0.0, abs_tol=atol),
        "saved_z_score": saved,
        "recomputed_z_score": recomputed,
        "abs_error": abs_error,
        "invalid": result["invalid"],
    }


__all__ = [
    "DEFAULT_TOKENIZER",
    "WllmConfigScorer",
    "check_saved_score",
    "load_tokenizer",
    "score_record",
    "score_wllm",
    "tokenize_for_detection",
    "tokenize_segment",
]
