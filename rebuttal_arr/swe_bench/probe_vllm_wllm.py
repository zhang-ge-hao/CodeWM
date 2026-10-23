#!/usr/bin/env python3
"""End-to-end token-level probe for the vLLM WLLM adapter."""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import torch
from transformers import AutoTokenizer

from src._sweet import WatermarkDetector


def request_completion(base_url: str, *, delta: float, vocab_size: int) -> dict:
    payload = {
        "model": "Qwen3.6-35B-A3B",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Write a Python function that computes all prime numbers "
                    "below n. Return only the code."
                ),
            }
        ],
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "max_tokens": 256,
        "seed": 23,
        "return_token_ids": True,
        "vllm_xargs": {
            "wllm_enabled": 1,
            "wllm_gamma": 0.5,
            "wllm_delta": delta,
            "wllm_ngram_len": 5,
            "wllm_hash_key": 15485863,
            "wllm_vocab_size": vocab_size,
            "wllm_debug": 1,
        },
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=900) as response:
        return json.load(response)


def detect(token_ids: list[int], tokenizer, *, delta: float) -> dict:
    detector = WatermarkDetector(
        vocab=list(tokenizer.get_vocab().values()),
        gamma=0.5,
        delta=delta,
        seeding_scheme="n_grams",
        ngram_len=5,
        hash_key=15485863,
        select_green_tokens=True,
        tokenizer=tokenizer,
        z_threshold=4.0,
    )
    result = detector.detect(
        tokenized_text=torch.tensor(token_ids, dtype=torch.long),
        tokenized_prefix=torch.tensor([], dtype=torch.long),
        return_green_token_mask=False,
    )
    return {
        key: value.item() if hasattr(value, "item") else value
        for key, value in result.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3.6-35B-A3B")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, local_files_only=True)
    vocab_size = len(tokenizer.get_vocab())
    results = []
    for delta in (4.0, 100.0):
        response = request_completion(
            args.base_url, delta=delta, vocab_size=vocab_size
        )
        choice = response["choices"][0]
        # vLLM's extension follows ChatCompletionResponseChoice, not the
        # nested OpenAI ChatMessage schema.
        token_ids = choice["token_ids"]
        results.append(
            {
                "delta": delta,
                "text": choice["message"].get("content"),
                "finish_reason": choice.get("finish_reason"),
                "token_ids": token_ids,
                "detection": detect(token_ids, tokenizer, delta=delta),
            }
        )
    output = {
        "model": args.model_id,
        "gamma": 0.5,
        "ngram_len": 5,
        "hash_key": 15485863,
        "vocab_size": vocab_size,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
