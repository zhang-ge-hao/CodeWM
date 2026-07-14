#!/usr/bin/env python3
"""Benchmark vLLM with real mini-SWE-agent conversation prefixes."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
import httpx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--trajectory-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=("initial", "long"), required=True)
    parser.add_argument("--concurrency", type=int, nargs="+", required=True)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0x1352766)
    parser.add_argument("--wllm", action="store_true")
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--delta", type=float, default=4.0)
    parser.add_argument("--watermark-key", type=int, default=15485863)
    parser.add_argument("--vocab-size", type=int, default=248077)
    parser.add_argument("--reset-prefix-cache-between-levels", action="store_true")
    return parser.parse_args()


def clean_message(message: dict[str, Any]) -> dict[str, str] | None:
    role = message.get("role")
    content = message.get("content")
    if role not in {"system", "user", "assistant"} or not isinstance(content, str):
        return None
    return {"role": role, "content": content}


def load_agent_templates(root: Path) -> tuple[str, str]:
    for path in sorted(root.glob("**/*.traj.json")):
        try:
            config = json.loads(path.read_text())["info"]["config"]["agent"]
            return str(config["system_template"]), str(config["instance_template"])
        except (OSError, KeyError, TypeError, json.JSONDecodeError):
            continue
    raise RuntimeError(f"No agent templates found below {root}")


def load_initial_prompts_from_selection(
    root: Path, selection_path: Path
) -> list[dict[str, Any]]:
    from datasets import load_dataset

    selection = json.loads(selection_path.read_text())
    instance_ids = selection.get("instance_ids") or [
        case["instance_id"] for case in selection["cases"]
    ]
    dataset_name = selection.get("dataset", "SWE-bench/SWE-bench_Verified")
    split = selection.get("split", "test")
    rows = load_dataset(dataset_name, split=split)
    by_id = {row["instance_id"]: row for row in rows}
    system_template, instance_template = load_agent_templates(root)
    prompts: list[dict[str, Any]] = []
    for instance_id in instance_ids:
        row = by_id[instance_id]
        messages = [
            {"role": "system", "content": system_template},
            {
                "role": "user",
                "content": instance_template.replace(
                    "{{task}}", str(row["problem_statement"])
                ),
            },
        ]
        prompts.append(
            {
                "instance_id": instance_id,
                "messages": messages,
                "characters": sum(len(msg["content"]) for msg in messages),
                "message_count": len(messages),
                "source": str(selection_path),
            }
        )
    return prompts


def load_prompts(
    root: Path, profile: str, selection_path: Path | None = None
) -> list[dict[str, Any]]:
    if profile == "initial" and selection_path is not None:
        return load_initial_prompts_from_selection(root, selection_path)

    prompts: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in sorted(root.glob("**/*.traj.json")):
        try:
            payload = json.loads(path.read_text())
            raw_messages = payload["messages"]
        except (OSError, KeyError, TypeError, json.JSONDecodeError):
            continue
        instance_id = path.stem.removesuffix(".traj")
        if instance_id in seen or not isinstance(raw_messages, list):
            continue
        messages = [item for raw in raw_messages if (item := clean_message(raw))]
        if len(messages) < 2:
            continue
        endpoints = (
            [2]
            if profile == "initial"
            else [i + 1 for i, msg in enumerate(messages) if msg["role"] == "user"]
        )
        for endpoint in endpoints:
            prefix = messages[:endpoint]
            prompts.append(
                {
                    "instance_id": (
                        instance_id
                        if profile == "initial"
                        else f"{instance_id}@messages-{endpoint}"
                    ),
                    "messages": prefix,
                    "characters": sum(len(msg["content"]) for msg in prefix),
                    "message_count": len(prefix),
                    "source": str(path),
                }
            )
        seen.add(instance_id)
    if not prompts:
        raise RuntimeError(f"No usable trajectories below {root}")
    # Exercise a range of repositories/context sizes instead of repeating only
    # the first case. Largest contexts are first for the long-context stress run.
    prompts.sort(key=lambda item: item["characters"], reverse=profile == "long")
    return prompts


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


async def one_request(
    client: AsyncOpenAI,
    args: argparse.Namespace,
    prompt: dict[str, Any],
    request_index: int,
) -> dict[str, Any]:
    extra_body: dict[str, Any] = {"top_k": args.top_k}
    if args.wllm:
        extra_body["vllm_xargs"] = {
            "wllm_enabled": 1,
            "wllm_gamma": args.gamma,
            "wllm_delta": args.delta,
            "wllm_ngram_len": 5,
            "wllm_hash_key": args.watermark_key,
            "wllm_vocab_size": args.vocab_size,
        }
    started = time.perf_counter()
    try:
        response = await client.chat.completions.create(
            model=args.model,
            messages=prompt["messages"],
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            seed=args.seed + request_index,
            extra_body=extra_body,
            timeout=1800,
        )
        elapsed = time.perf_counter() - started
        usage = response.usage
        output_tokens = int(usage.completion_tokens if usage else 0)
        input_tokens = int(usage.prompt_tokens if usage else 0)
        finish_reason = response.choices[0].finish_reason if response.choices else None
        return {
            "status": "ok",
            "instance_id": prompt["instance_id"],
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "elapsed_seconds": elapsed,
            "output_tokens_per_second": output_tokens / elapsed if elapsed else None,
            "finish_reason": finish_reason,
            "prompt_characters": prompt["characters"],
            "message_count": prompt["message_count"],
        }
    except Exception as exc:
        return {
            "status": "error",
            "instance_id": prompt["instance_id"],
            "elapsed_seconds": time.perf_counter() - started,
            "error": f"{type(exc).__name__}: {exc}",
            "prompt_characters": prompt["characters"],
            "message_count": prompt["message_count"],
        }


async def run_level(
    client: AsyncOpenAI,
    args: argparse.Namespace,
    prompts: list[dict[str, Any]],
    concurrency: int,
) -> dict[str, Any]:
    selected = [prompts[i % len(prompts)] for i in range(concurrency)]
    started = time.perf_counter()
    rows = await asyncio.gather(
        *(one_request(client, args, prompt, i) for i, prompt in enumerate(selected))
    )
    makespan = time.perf_counter() - started
    good = [row for row in rows if row["status"] == "ok"]
    rates = [float(row["output_tokens_per_second"]) for row in good]
    total_output = sum(int(row["output_tokens"]) for row in good)
    return {
        "concurrency": concurrency,
        "profile": args.profile,
        "wllm": args.wllm,
        "successful": len(good),
        "failed": len(rows) - len(good),
        "makespan_seconds": makespan,
        "total_output_tokens": total_output,
        "aggregate_output_tokens_per_second": total_output / makespan if makespan else None,
        "per_request_rate_min": min(rates) if rates else None,
        "per_request_rate_p10": percentile(rates, 0.10),
        "per_request_rate_median": statistics.median(rates) if rates else None,
        "per_request_rate_max": max(rates) if rates else None,
        "requests": rows,
    }


async def async_main(args: argparse.Namespace) -> None:
    prompts = load_prompts(args.trajectory_root, args.profile, args.selection)
    client = AsyncOpenAI(base_url=args.base_url.rstrip("/") + "/v1", api_key="EMPTY")
    await one_request(client, args, prompts[0], -1)  # compile/cache warm-up
    levels = []
    for concurrency in args.concurrency:
        if args.reset_prefix_cache_between_levels:
            async with httpx.AsyncClient(timeout=60) as cache_client:
                response = await cache_client.post(
                    args.base_url.rstrip("/") + "/reset_prefix_cache"
                )
                response.raise_for_status()
        result = await run_level(client, args, prompts, concurrency)
        levels.append(result)
        print(json.dumps({key: value for key, value in result.items() if key != "requests"}), flush=True)
    payload = {
        "model": args.model,
        "base_url": args.base_url,
        "profile": args.profile,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "wllm": args.wllm,
        "available_unique_prompts": len(prompts),
        "levels": levels,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    await client.close()


def main() -> None:
    asyncio.run(async_main(parse_args()))


if __name__ == "__main__":
    main()
