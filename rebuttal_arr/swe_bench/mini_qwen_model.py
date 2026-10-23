"""Shared Hugging Face Qwen adapter for mini-swe-agent text actions.

The agent controller and model stay in the Slurm process.  Only shell actions
are sent to the mini-swe-agent SWE-ReX Modal environment.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from typing import Any

from pydantic import BaseModel


LOGGER = logging.getLogger("mini_qwen_model")


class QwenHFTextModelConfig(BaseModel):
    model_name: str = "Qwen/Qwen3.6-35B-A3B"
    watermarking: str = "none"
    gamma: float | None = None
    delta: float | None = None
    ngram_len: int = 5
    watermark_key: int = 15485863
    generation_seed: int = 0x1352766
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 20
    max_context_tokens: int = 200_000
    max_new_tokens: int = 32_768
    max_generation_seconds: float = 300.0
    local_files_only: bool = True
    action_regex: str = r"```mswea_bash_command\s*\n(.*?)\n```"
    format_error_template: str = (
        "Please always provide EXACTLY ONE action in triple backticks, "
        "found {{actions|length}} actions."
    )
    observation_template: str = (
        "{% if output.exception_info %}<exception>{{output.exception_info}}</exception>\n{% endif %}"
        "<returncode>{{output.returncode}}</returncode>\n<output>\n{{output.output}}</output>"
    )
    multimodal_regex: str = ""


class _SharedBackend:
    """One model copy and one generation stream shared by all agent workers."""

    load_lock = threading.Lock()
    generation_lock = threading.Lock()
    model = None
    tokenizer = None
    model_name: str | None = None
    local_files_only: bool | None = None
    device_map: dict[str, str] | None = None

    @classmethod
    def get(cls, config: QwenHFTextModelConfig):
        with cls.load_lock:
            if cls.model is None:
                cls._load(config)
            elif (
                cls.model_name != config.model_name
                or cls.local_files_only != config.local_files_only
            ):
                raise RuntimeError("A different Qwen backend is already loaded")
        return cls.model, cls.tokenizer

    @classmethod
    def _load(cls, config: QwenHFTextModelConfig) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
            raise RuntimeError(
                "Qwen mini-swe-agent runs require exactly two visible CUDA devices"
            )
        for index in range(2):
            memory = torch.cuda.get_device_properties(index).total_memory
            if memory < 75 * 1024**3:
                raise RuntimeError(
                    f"cuda:{index} has only {memory / 1024**3:.1f} GiB; "
                    "two 80-GiB A100s are required"
                )

        LOGGER.info("Loading %s once across two GPUs", config.model_name)
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            local_files_only=config.local_files_only,
        )
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            dtype=torch.bfloat16,
            device_map="balanced",
            max_memory={0: "72GiB", 1: "72GiB", "cpu": "96GiB"},
            low_cpu_mem_usage=True,
            local_files_only=config.local_files_only,
        ).eval()
        device_map = {key: str(value) for key, value in model.hf_device_map.items()}
        placements = set(device_map.values())
        normalized = {
            f"cuda:{value}" if value.isdigit() else value for value in placements
        }
        if "cpu" in normalized or "disk" in normalized:
            raise RuntimeError(f"Model was offloaded outside GPUs: {normalized}")
        if not {"cuda:0", "cuda:1"}.issubset(normalized):
            raise RuntimeError(f"Model is not distributed across both GPUs: {normalized}")

        cls.model = model
        cls.tokenizer = tokenizer
        cls.model_name = config.model_name
        cls.local_files_only = config.local_files_only
        cls.device_map = device_map
        LOGGER.info("Qwen loaded with placements=%s", sorted(normalized))


def _stable_call_seed(messages: list[dict[str, Any]], base_seed: int) -> int:
    digest = hashlib.sha256()
    digest.update(str(base_seed).encode())
    for message in messages:
        digest.update(str(message.get("role", "")).encode())
        digest.update(b"\0")
        digest.update(str(message.get("content", "")).encode())
        digest.update(b"\0")
    return int.from_bytes(digest.digest()[:8], "big") & 0x7FFFFFFF


def _watermark_logits_processor(tokenizer, config: QwenHFTextModelConfig):
    if config.watermarking == "none":
        return None
    if config.watermarking != "wllm":
        raise ValueError(f"Unsupported watermarking mode: {config.watermarking}")
    if config.gamma is None or config.delta is None:
        raise ValueError("WLLM requires gamma and delta")

    from transformers import LogitsProcessorList
    from src._sweet import WatermarkLogitsProcessor

    processor = WatermarkLogitsProcessor(
        vocab=list(tokenizer.get_vocab().values()),
        gamma=config.gamma,
        delta=config.delta,
        seeding_scheme="n_grams",
        ngram_len=config.ngram_len,
        hash_key=config.watermark_key,
        select_green_tokens=True,
    )
    return LogitsProcessorList([processor])


class QwenHFTextModel:
    """mini-swe-agent model interface using official backtick-style actions."""

    def __init__(self, **kwargs: Any) -> None:
        self.config = QwenHFTextModelConfig(**kwargs)
        if self.config.max_new_tokens < 10_000:
            raise ValueError("max_new_tokens must be at least 10,000")
        if not 0 < self.config.max_context_tokens <= 262_144:
            raise ValueError("max_context_tokens must be in (0, 262144]")
        if self.config.watermarking == "wllm":
            if self.config.gamma is None or not 0 < self.config.gamma < 1:
                raise ValueError("WLLM gamma must be in (0, 1)")
            if self.config.delta is None or self.config.delta <= 0:
                raise ValueError("WLLM delta must be positive")
        self.model, self.tokenizer = _SharedBackend.get(self.config)
        self.logits_processor = _watermark_logits_processor(
            self.tokenizer, self.config
        )

    def query(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        import torch
        from minisweagent.models.utils.actions_text import parse_regex_actions
        from transformers import set_seed

        prepared = [
            {"role": message["role"], "content": message.get("content", "")}
            for message in messages
        ]
        with _SharedBackend.generation_lock:
            rendered = self.tokenizer.apply_chat_template(
                prepared,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
                preserve_thinking=True,
            )
            encoded = self.tokenizer(
                rendered,
                add_special_tokens=False,
                return_tensors="pt",
            )
            input_tokens = int(encoded["input_ids"].shape[1])
            remaining = self.config.max_context_tokens - input_tokens
            if remaining <= 0:
                raise RuntimeError(
                    f"Agent history has {input_tokens} tokens, exceeding the "
                    f"{self.config.max_context_tokens}-token context limit"
                )
            max_new_tokens = min(self.config.max_new_tokens, remaining)
            input_device = self.model.get_input_embeddings().weight.device
            encoded = encoded.to(input_device)
            call_seed = _stable_call_seed(prepared, self.config.generation_seed)
            set_seed(call_seed, deterministic=False)

            started = time.monotonic()
            with torch.inference_mode():
                sequences = self.model.generate(
                    **encoded,
                    do_sample=True,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    top_k=self.config.top_k,
                    max_new_tokens=max_new_tokens,
                    max_time=self.config.max_generation_seconds,
                    use_cache=True,
                    logits_processor=self.logits_processor,
                )
            for index in range(torch.cuda.device_count()):
                torch.cuda.synchronize(index)
            elapsed = time.monotonic() - started
            output_ids = sequences[0, input_tokens:].detach().cpu().tolist()
            content = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            del sequences, encoded

        actions = parse_regex_actions(
            content,
            action_regex=self.config.action_regex,
            format_error_template=self.config.format_error_template,
        )
        output_tokens = len(output_ids)
        LOGGER.info(
            "Generated agent turn: input=%d output=%d elapsed=%.1fs rate=%.2f tok/s",
            input_tokens,
            output_tokens,
            elapsed,
            output_tokens / elapsed if elapsed else 0.0,
        )
        return {
            "role": "assistant",
            "content": content,
            "extra": {
                "actions": actions,
                "cost": 0.0,
                "timestamp": time.time(),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "elapsed_seconds": round(elapsed, 3),
                "generation_seed": call_seed,
                "hit_max_new_tokens": output_tokens >= max_new_tokens,
                "hit_max_generation_time": (
                    elapsed >= self.config.max_generation_seconds
                ),
                "watermarking": self.config.watermarking,
            },
        }

    def format_message(self, **kwargs: Any) -> dict[str, Any]:
        from minisweagent.models.utils.openai_multimodal import (
            expand_multimodal_content,
        )

        return expand_multimodal_content(kwargs, pattern=self.config.multimodal_regex)

    def format_observation_messages(
        self,
        message: dict[str, Any],
        outputs: list[dict[str, Any]],
        template_vars: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        from minisweagent.models.utils.actions_text import (
            format_observation_messages,
        )

        return format_observation_messages(
            outputs,
            observation_template=self.config.observation_template,
            template_vars=template_vars,
            multimodal_regex=self.config.multimodal_regex,
        )

    def get_template_vars(self, **kwargs: Any) -> dict[str, Any]:
        return self.config.model_dump()

    def serialize(self) -> dict[str, Any]:
        config = self.config.model_dump(mode="json")
        return {
            "info": {
                "config": {
                    "model": config,
                    "model_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                    "device_map": _SharedBackend.device_map,
                }
            }
        }
