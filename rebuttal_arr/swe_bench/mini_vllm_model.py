"""mini-SWE-agent model adapter for a local vLLM OpenAI server."""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Any

from minisweagent.models.litellm_model import LitellmModel


LOGGER = logging.getLogger("mini_vllm_model")


def stable_call_seed(messages: list[dict[str, Any]], base_seed: int) -> int:
    digest = hashlib.sha256(str(base_seed).encode())
    for message in messages:
        digest.update(str(message.get("role", "")).encode())
        digest.update(b"\0")
        digest.update(str(message.get("content", "")).encode())
        digest.update(b"\0")
    return int.from_bytes(digest.digest()[:8], "big") & 0x7FFFFFFF


class LoggedVLLMModel(LitellmModel):
    """Use mini-SWE-agent's official tool-call parser with local vLLM."""

    def __init__(self, *, generation_seed: int, **kwargs: Any) -> None:
        self.generation_seed = int(generation_seed)
        super().__init__(**kwargs)

    def _query(self, messages: list[dict[str, str]], **kwargs: Any):
        kwargs.setdefault("seed", stable_call_seed(messages, self.generation_seed))
        return super()._query(messages, **kwargs)

    def query(self, messages: list[dict[str, str]], **kwargs: Any) -> dict:
        started = time.monotonic()
        message = super().query(messages, **kwargs)
        elapsed = time.monotonic() - started
        response = message.get("extra", {}).get("response", {})
        usage = response.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        output_tokens = int(usage.get("completion_tokens") or 0)
        message["extra"].update(
            {
                "elapsed_seconds": round(elapsed, 3),
                "input_tokens": prompt_tokens,
                "output_tokens": output_tokens,
            }
        )
        LOGGER.info(
            "vLLM agent turn: input=%d output=%d elapsed=%.1fs rate=%.2f tok/s",
            prompt_tokens,
            output_tokens,
            elapsed,
            output_tokens / elapsed if elapsed else 0.0,
        )
        return message
