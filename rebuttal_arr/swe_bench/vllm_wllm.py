"""vLLM adapter for the WLLM n-gram watermark logits processor.

The request-level implementation intentionally matches ``src._sweet``:
SHA3-512 hashes the preceding n-gram, a CPU ``torch.Generator`` produces the
vocabulary permutation, and ``delta`` is added to the selected green tokens.
The vLLM adapter keeps one request-level processor per active request so
continuous batching cannot mix watermark state between agent trajectories.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor import (
    AdapterLogitsProcessor,
    RequestLogitsProcessor,
)

LOGGER = logging.getLogger("vllm_wllm")


class WLLMRequestLogitsProcessor:
    """Apply the existing WLLM rule to one vLLM request."""

    def __init__(
        self,
        *,
        gamma: float,
        delta: float,
        ngram_len: int,
        hash_key: int,
        vocab_size: int | None = None,
        debug: bool = False,
    ) -> None:
        if not 0.0 < gamma < 1.0:
            raise ValueError("gamma must be in (0, 1)")
        if delta <= 0.0:
            raise ValueError("delta must be positive")
        if ngram_len < 2:
            raise ValueError("ngram_len must be at least 2")
        self.gamma = float(gamma)
        self.delta = float(delta)
        self.ngram_len = int(ngram_len)
        self.hash_key = int(hash_key)
        self.vocab_size = int(vocab_size) if vocab_size is not None else None
        if self.vocab_size is not None and self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        self.rng = torch.Generator(device="cpu")
        self.calls = 0
        self.debug = bool(debug)
        self.previous_green_ids: torch.Tensor | None = None

    @staticmethod
    def _hash_integer_array_to_int(values: list[int]) -> int:
        encoded = ",".join(map(str, values)).encode()
        return int(hashlib.sha3_512(encoded).hexdigest(), 16) & 0xFFFFFFFFFFFFFFFF

    def __call__(
        self,
        prompt_token_ids: list[int],
        output_token_ids: list[int],
        logits: torch.Tensor,
    ) -> torch.Tensor:
        prefix = prompt_token_ids + output_token_ids
        required = self.ngram_len - 1
        if len(prefix) < required:
            return logits

        # src._sweet iterates backwards over the preceding tokens. Preserve
        # that ordering because it is part of the detector-compatible seed.
        previous = [prefix[-index] for index in range(1, self.ngram_len)]
        seed = self._hash_integer_array_to_int(previous) ^ self.hash_key
        self.rng.manual_seed(seed)

        # Model output tensors may include padding entries beyond the actual
        # tokenizer vocabulary (Qwen3.6: 248320 logits vs 248077 tokens). The
        # existing WLLM detector permutes len(tokenizer.get_vocab()), so using
        # the padded logits width here would produce a different greenlist.
        vocab_size = self.vocab_size or int(logits.shape[-1])
        if vocab_size > int(logits.shape[-1]):
            raise ValueError("WLLM vocabulary exceeds logits width")
        greenlist_size = int(vocab_size * self.gamma)
        green_ids = torch.randperm(
            vocab_size,
            generator=self.rng,
            device="cpu",
        )[:greenlist_size]
        logits[green_ids.to(device=logits.device)] += self.delta
        if self.debug and self.calls < 8:
            previous_token_green = None
            if output_token_ids and self.previous_green_ids is not None:
                previous_token_green = bool(
                    (self.previous_green_ids == output_token_ids[-1]).any().item()
                )
            LOGGER.warning(
                "WLLM active: prompt_tokens=%d output_tokens=%d logits=%d "
                "vocab=%d gamma=%g delta=%g ngram=%d output_tail=%s "
                "previous_token_green=%s",
                len(prompt_token_ids),
                len(output_token_ids),
                int(logits.shape[-1]),
                vocab_size,
                self.gamma,
                self.delta,
                self.ngram_len,
                output_token_ids[-8:],
                previous_token_green,
            )
        if self.debug:
            self.previous_green_ids = green_ids
        self.calls += 1
        return logits


class VLLMWatermarkLogitsProcessor(AdapterLogitsProcessor):
    """Enable WLLM per request through vLLM ``vllm_xargs``."""

    KEYS = {
        "wllm_enabled",
        "wllm_gamma",
        "wllm_delta",
        "wllm_ngram_len",
        "wllm_hash_key",
        "wllm_vocab_size",
    }

    @staticmethod
    def _enabled(value: Any) -> bool:
        # OpenAI-compatible ``vllm_xargs`` are normalized to strings by
        # vLLM, even when the JSON value was a boolean.
        if isinstance(value, bool):
            return value
        # Chat-completions declares xarg values as str|int|float (not bool),
        # so Pydantic normalizes JSON true/false to 1/0 in vLLM 0.19.
        if isinstance(value, (int, float)) and value in {0, 1}:
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes", "on"}:
                return True
            if normalized in {"false", "0", "no", "off", ""}:
                return False
        raise ValueError("wllm_enabled must be a boolean value")

    @classmethod
    def validate_params(cls, params: SamplingParams) -> None:
        values = params.extra_args or {}
        enabled = cls._enabled(values.get("wllm_enabled", False))
        if not enabled:
            return
        missing = cls.KEYS - values.keys()
        if missing:
            raise ValueError(f"Missing WLLM request arguments: {sorted(missing)}")
        WLLMRequestLogitsProcessor(
            gamma=float(values["wllm_gamma"]),
            delta=float(values["wllm_delta"]),
            ngram_len=int(values["wllm_ngram_len"]),
            hash_key=int(values["wllm_hash_key"]),
            vocab_size=int(values["wllm_vocab_size"]),
            debug=cls._enabled(values.get("wllm_debug", False)),
        )

    def is_argmax_invariant(self) -> bool:
        return False

    def new_req_logits_processor(
        self, params: SamplingParams
    ) -> RequestLogitsProcessor | None:
        self.validate_params(params)
        values: dict[str, Any] = params.extra_args or {}
        if not self._enabled(values.get("wllm_enabled", False)):
            return None
        return WLLMRequestLogitsProcessor(
            gamma=float(values["wllm_gamma"]),
            delta=float(values["wllm_delta"]),
            ngram_len=int(values["wllm_ngram_len"]),
            hash_key=int(values["wllm_hash_key"]),
            vocab_size=int(values["wllm_vocab_size"]),
            debug=self._enabled(values.get("wllm_debug", False)),
        )
