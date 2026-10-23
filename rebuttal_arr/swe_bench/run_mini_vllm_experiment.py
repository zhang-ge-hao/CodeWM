#!/usr/bin/env python3
"""Run concurrent mini-SWE-agent cases against a local vLLM server."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from mini_vllm_model import LoggedVLLMModel
from run_experiment import (
    aggregate_official_reports,
    evaluate_one_modal,
    prepare_async_evaluation,
)
from run_mini_experiment import (
    NGRAM_LEN,
    atomic_write_json,
    atomic_write_jsonl,
    append_jsonl,
    configure_logging,
    create_deadline_agent,
    create_patch_detector,
    detect_added_code,
    load_cases,
    summarize,
)


MODEL_ID = "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4"
AGENT_WALL_TIME_LIMIT_SECONDS = 1800.0
MODAL_SANDBOX_CLEANUP_GRACE_SECONDS = 300.0
SCHEMA_VERSION = 2
LOGGER = logging.getLogger("mini_swe_agent_vllm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--selection-source", type=Path)
    parser.add_argument("--generation-seed", type=int, default=0x1352766)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument(
        "--served-model-name", default="Qwen3.5-35B-A3B-GPTQ-Int4"
    )
    parser.add_argument("--vllm-base-url", required=True)
    parser.add_argument("--watermarking", choices=("none", "wllm"), default="none")
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--delta", type=float)
    parser.add_argument("--watermark-key", type=int, default=15485863)
    parser.add_argument("--z-threshold", type=float, default=4.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=32_768)
    parser.add_argument("--agent-workers", type=int, default=4)
    parser.add_argument(
        "--environment-start-attempts",
        type=int,
        default=5,
        help="Retry only transient Modal SandboxTimeoutError failures.",
    )
    parser.add_argument(
        "--agent-wall-time-limit-seconds",
        type=float,
        default=AGENT_WALL_TIME_LIMIT_SECONDS,
    )
    parser.add_argument("--eval-workers", type=int, default=8)
    parser.add_argument("--modal-eval-timeout", type=int, default=1800)
    parser.add_argument("--swebench-python", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--skip-evaluation", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.sample_size <= 500:
        parser.error("--sample-size must be in [1, 500]")
    if args.watermarking == "wllm":
        if args.gamma is None or not 0 < args.gamma < 1:
            parser.error("WLLM requires --gamma in (0, 1)")
        if args.delta is None or args.delta <= 0:
            parser.error("WLLM requires positive --delta")
    if args.agent_workers <= 0 or args.eval_workers <= 0:
        parser.error("worker counts must be positive")
    if args.environment_start_attempts <= 0:
        parser.error("--environment-start-attempts must be positive")
    if args.agent_wall_time_limit_seconds <= 0:
        parser.error("--agent-wall-time-limit-seconds must be positive")
    if not args.swebench_python.is_file():
        parser.error(f"SWE-bench Python does not exist: {args.swebench_python}")
    if args.selection_source and not args.selection_source.is_file():
        parser.error(f"Selection source does not exist: {args.selection_source}")
    return args


def load_official_config() -> dict[str, Any]:
    from minisweagent.config import builtin_config_dir, get_config_from_spec
    from minisweagent.utils.serialize import recursive_merge

    benchmark_dir = builtin_config_dir / "benchmarks"
    base = get_config_from_spec(str(benchmark_dir / "swebench.yaml"))
    modal = get_config_from_spec(str(benchmark_dir / "swebench_modal.yaml"))
    config = recursive_merge(base, modal)
    if config["agent"].get("step_limit") != 250:
        raise RuntimeError("Official mini-SWE-agent step_limit is not 250")
    if config["environment"].get("timeout") != 60:
        raise RuntimeError("Official mini-SWE-agent command timeout is not 60 seconds")
    if config["environment"].get("runtime_timeout") != 1800.0:
        raise RuntimeError("Official Modal runtime timeout is not 1800 seconds")
    return config


def ensure_modal_sandbox_outlives_agent(
    config: dict[str, Any], agent_wall_time_limit_seconds: float
) -> None:
    required_lifetime = (
        float(agent_wall_time_limit_seconds)
        + MODAL_SANDBOX_CLEANUP_GRACE_SECONDS
    )
    environment = config["environment"]
    environment["deployment_timeout"] = max(
        float(environment.get("deployment_timeout", 0.0)),
        required_lifetime,
    )


def experiment_name(args: argparse.Namespace) -> str:
    if args.watermarking == "none":
        return "non-wm"
    return f"wllm-delta-{args.delta:g}-gamma-{args.gamma:g}-ngram-{NGRAM_LEN}"


def prediction_model_name(args: argparse.Namespace) -> str:
    return (
        f"{args.served_model_name}--mini-swe-agent-v2.2.6-vllm--"
        f"{experiment_name(args)}"
    )


def make_model(
    args: argparse.Namespace, config: dict[str, Any], *, vocab_size: int
) -> LoggedVLLMModel:
    extra_body: dict[str, Any] = {
        "top_k": args.top_k,
        "chat_template_kwargs": {
            "enable_thinking": True,
            "preserve_thinking": True,
        },
    }
    if args.watermarking == "wllm":
        extra_body["vllm_xargs"] = {
            "wllm_enabled": 1,
            "wllm_gamma": args.gamma,
            "wllm_delta": args.delta,
            "wllm_ngram_len": NGRAM_LEN,
            "wllm_hash_key": args.watermark_key,
            "wllm_vocab_size": vocab_size,
        }
    model_config = config["model"]
    return LoggedVLLMModel(
        generation_seed=args.generation_seed,
        model_name=f"openai/{args.served_model_name}",
        cost_tracking="ignore_errors",
        format_error_template=model_config["format_error_template"],
        observation_template=model_config["observation_template"],
        multimodal_regex=model_config.get("multimodal_regex", ""),
        model_kwargs={
            "api_base": args.vllm_base_url.rstrip("/") + "/v1",
            "api_key": "EMPTY",
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_new_tokens,
            "drop_params": True,
            "extra_body": extra_body,
        },
    )


def get_sb_environment_with_retries(
    config: dict[str, Any],
    case: dict[str, Any],
    *,
    attempts: int,
    index: int,
    total: int,
):
    from minisweagent.run.benchmarks.swebench import get_sb_environment
    from modal.exception import SandboxTimeoutError

    for attempt in range(1, attempts + 1):
        try:
            return get_sb_environment(config, case)
        except SandboxTimeoutError:
            if attempt == attempts:
                raise
            LOGGER.warning(
                "[%d/%d] Modal sandbox startup timed out for %s; "
                "retrying (%d/%d)",
                index,
                total,
                case["instance_id"],
                attempt + 1,
                attempts,
            )
    raise RuntimeError("Modal environment creation exhausted its attempts")


def stop_sb_environment(env) -> None:
    deployment = getattr(env, "deployment", None)
    sandbox = getattr(deployment, "_sandbox", None)
    sandbox_id = getattr(sandbox, "object_id", None)
    env.stop()
    if not sandbox_id:
        return
    try:
        from modal import Sandbox

        Sandbox.from_id(sandbox_id).terminate(wait=False)
    except Exception as exc:
        LOGGER.warning(
            "Could not explicitly terminate Modal sandbox %s: %s",
            sandbox_id,
            exc,
        )


def run_agent_case(
    *,
    case: dict[str, Any],
    index: int,
    total: int,
    data_dir: Path,
    config: dict[str, Any],
    model: LoggedVLLMModel,
    detector,
    tokenizer,
    args: argparse.Namespace,
) -> dict[str, Any]:
    instance_id = case["instance_id"]
    case_dir = data_dir / "trajectories" / f"case-{index:03d}-{instance_id}"
    case_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = case_dir / f"{instance_id}.traj.json"
    LOGGER.info("[%d/%d] Starting official mini-SWE-agent for %s", index, total, instance_id)
    started = time.monotonic()
    env = None
    agent = None
    info: dict[str, Any] = {}
    error: str | None = None
    try:
        env = get_sb_environment_with_retries(
            config,
            case,
            attempts=args.environment_start_attempts,
            index=index,
            total=total,
        )
        agent_config = {
            **config["agent"],
            "wall_time_limit_seconds": args.agent_wall_time_limit_seconds,
        }
        agent = create_deadline_agent(
            model,
            env,
            **agent_config,
            output_path=trajectory_path,
        )
        info = agent.run(case["problem_statement"])
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        LOGGER.exception("[%d/%d] Agent failed for %s", index, total, instance_id)
        info = {"exit_status": type(exc).__name__, "submission": ""}
    finally:
        if agent is not None:
            agent.save(
                trajectory_path,
                {"instance_id": instance_id, "info": {"runner_error": error}},
            )
        if env is not None:
            stop_sb_environment(env)

    patch = info.get("submission") or ""
    added_code, detection = detect_added_code(
        patch, detector, tokenizer, args
    )
    (case_dir / "patch.diff").write_text(patch)
    (case_dir / "added_code.txt").write_text(added_code)
    row = {
        "schema_version": SCHEMA_VERSION,
        "status": "ok" if error is None else "error",
        "instance_id": instance_id,
        "model_name_or_path": prediction_model_name(args),
        "model_patch": patch,
        "exit_status": info.get("exit_status"),
        "runner_error": error,
        "deadline_forced_submission": bool(
            agent is not None
            and getattr(agent, "deadline_forced_submission", False)
        ),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "patch_characters": len(patch),
        "added_code_characters": len(added_code),
        "watermarking": args.watermarking,
        "delta": args.delta,
        "gamma": args.gamma,
        "ngram_len": NGRAM_LEN,
        "detection": detection,
    }
    atomic_write_json(case_dir / "result.json", row)
    append_jsonl(data_dir / "case_results.jsonl", row)
    z_text = f" z={detection.get('z_score')}" if isinstance(detection, dict) else ""
    LOGGER.info(
        "[%d/%d] Agent result %s: exit=%s elapsed=%.1fs patch_chars=%d%s",
        index,
        total,
        instance_id,
        row["exit_status"],
        row["elapsed_seconds"],
        len(patch),
        z_text,
    )
    return row


def write_predictions(
    path: Path, selected_ids: list[str], rows: dict[str, dict[str, Any]]
) -> None:
    atomic_write_jsonl(
        path,
        (
            {
                "instance_id": instance_id,
                "model_name_or_path": rows[instance_id]["model_name_or_path"],
                "model_patch": rows[instance_id]["model_patch"],
            }
            for instance_id in selected_ids
            if instance_id in rows
        ),
    )


def main() -> None:
    args = parse_args()
    configure_logging(args.data_dir)
    mini_version = importlib.metadata.version("mini-swe-agent")
    if mini_version != "2.2.6":
        raise RuntimeError(f"Expected mini-swe-agent 2.2.6, found {mini_version}")
    cases, manifest = load_cases(args)
    config = load_official_config()
    ensure_modal_sandbox_outlives_agent(
        config, args.agent_wall_time_limit_seconds
    )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, local_files_only=True)
    tokenizer_vocab_size = len(tokenizer.get_vocab())
    model = make_model(args, config, vocab_size=tokenizer_vocab_size)
    detector = create_patch_detector(tokenizer, args)
    settings = {
        "schema_version": SCHEMA_VERSION,
        "model_id": args.model_id,
        "inference_backend": "vllm-0.19.0-openai-server",
        "vllm_base_url": args.vllm_base_url,
        "mini_swe_agent_version": mini_version,
        "official_configs": ["swebench.yaml", "swebench_modal.yaml"],
        "step_limit": config["agent"]["step_limit"],
        "agent_wall_time_limit_seconds": args.agent_wall_time_limit_seconds,
        "command_timeout": config["environment"]["timeout"],
        "modal_runtime_timeout": config["environment"]["runtime_timeout"],
        "modal_deployment_timeout": config["environment"]["deployment_timeout"],
        "modal_sandbox_cleanup_grace_seconds": MODAL_SANDBOX_CLEANUP_GRACE_SECONDS,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
        "agent_workers": args.agent_workers,
        "environment_start_attempts": args.environment_start_attempts,
        "watermarking": args.watermarking,
        "gamma": args.gamma,
        "delta": args.delta,
        "ngram_len": NGRAM_LEN,
        "watermark_key": args.watermark_key,
        "tokenizer_vocab_size": tokenizer_vocab_size,
        "detector_scope": "all_patch_added_lines",
        "selection": manifest,
    }
    atomic_write_json(args.data_dir / "settings.json", settings)

    rows: dict[str, dict[str, Any]] = {}
    eval_futures: dict[Future, str] = {}
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.eval_workers) as eval_pool:
        with ThreadPoolExecutor(max_workers=args.agent_workers) as agent_pool:
            agent_futures = {
                agent_pool.submit(
                    run_agent_case,
                    case=case,
                    index=index,
                    total=len(cases),
                    data_dir=args.data_dir,
                    config=config,
                    model=model,
                    detector=detector,
                    tokenizer=tokenizer,
                    args=args,
                ): (index, case["instance_id"])
                for index, case in enumerate(cases, 1)
            }
            for future in as_completed(agent_futures):
                index, instance_id = agent_futures[future]
                try:
                    row = future.result()
                except Exception as exc:
                    LOGGER.error("Uncaught agent worker error for %s: %s\n%s", instance_id, exc, traceback.format_exc())
                    row = {
                        "instance_id": instance_id,
                        "model_name_or_path": prediction_model_name(args),
                        "model_patch": "",
                        "status": "uncaught_error",
                        "runner_error": f"{type(exc).__name__}: {exc}",
                    }
                rows[instance_id] = row
                write_predictions(args.data_dir / "predictions.jsonl", manifest["instance_ids"], rows)
                if not args.skip_evaluation:
                    case_dir, evaluation_run_id = prepare_async_evaluation(
                        data_dir=args.data_dir,
                        base_run_id=args.run_id,
                        generation_index=index,
                        instance_id=instance_id,
                        row=row,
                    )
                    eval_future = eval_pool.submit(
                        evaluate_one_modal,
                        swebench_python=args.swebench_python,
                        case_dir=case_dir,
                        instance_id=instance_id,
                        evaluation_run_id=evaluation_run_id,
                        timeout=args.modal_eval_timeout,
                    )
                    eval_futures[eval_future] = instance_id
        for future in as_completed(eval_futures):
            instance_id = eval_futures[future]
            try:
                future.result()
            except Exception:
                LOGGER.exception("Final evaluator failed for %s", instance_id)

    official_report = None
    if not args.skip_evaluation:
        official_report = aggregate_official_reports(
            data_dir=args.data_dir,
            run_id=args.run_id,
            selected_ids=manifest["instance_ids"],
            prediction_name=prediction_model_name(args),
        )
    summary = summarize(
        data_dir=args.data_dir,
        args=args,
        selected_ids=manifest["instance_ids"],
        rows=rows,
        official_report=official_report,
    )
    summary["wall_elapsed_seconds"] = round(time.monotonic() - started, 3)
    summary["mean_agent_case_seconds"] = (
        sum(float(row.get("elapsed_seconds", 0.0)) for row in rows.values()) / len(rows)
        if rows
        else None
    )
    atomic_write_json(args.data_dir / "summary.json", summary)
    LOGGER.info("vLLM experiment summary: %s", json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
