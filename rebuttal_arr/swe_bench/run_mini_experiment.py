#!/usr/bin/env python3
"""Run Qwen with mini-swe-agent on SWE-bench Verified using Modal sandboxes."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import importlib.metadata
import json
import logging
import os
import random
import re
import statistics
import sys
import threading
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mini_qwen_model import QwenHFTextModel  # noqa: E402
from run_experiment import (  # noqa: E402
    GPUHeartbeat,
    aggregate_official_reports,
    evaluate_one_modal,
    ideal_auroc,
    prepare_async_evaluation,
)


VERIFIED_DATASET = "SWE-bench/SWE-bench_Verified"
MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
NGRAM_LEN = 5
SCHEMA_VERSION = 1
AGENT_WALL_TIME_LIMIT_SECONDS = 1800
FORCED_SUBMIT_COMMAND = (
    "cd /testbed && echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && "
    "if test -s patch.txt; then cat patch.txt; else git diff -- .; fi"
)
LOGGER = logging.getLogger("mini_swe_agent_experiment")
WRITE_LOCK = threading.Lock()
DETECT_LOCK = threading.Lock()


def create_deadline_agent(model, env, **kwargs):
    """Create an official DefaultAgent with a real best-effort deadline.

    mini-swe-agent 2.2.6's AgentConfig has no wall-time field. Extending its
    validated config avoids silently ignored kwargs. At the deadline, submit
    the current working-tree diff so a long-running case still yields a patch
    for official evaluation and watermark detection.
    """
    from minisweagent.agents.default import AgentConfig, DefaultAgent
    from minisweagent.exceptions import LimitsExceeded, Submitted

    class DeadlineAgentConfig(AgentConfig):
        wall_time_limit_seconds: float

    class DeadlineAgent(DefaultAgent):
        def __init__(self, *args, **agent_kwargs):
            super().__init__(
                *args,
                config_class=DeadlineAgentConfig,
                **agent_kwargs,
            )
            self.deadline_forced_submission = False
            self._deadline_at: float | None = None

        def run(self, task: str = "", **run_kwargs) -> dict:
            self._deadline_at = (
                time.monotonic() + self.config.wall_time_limit_seconds
            )
            return super().run(task, **run_kwargs)

        def _deadline_expired(self) -> bool:
            return (
                self._deadline_at is not None
                and time.monotonic() >= self._deadline_at
            )

        def _submit_at_deadline(self):
            self.deadline_forced_submission = True
            LOGGER.warning(
                "Agent wall-time deadline reached; submitting current diff"
            )
            try:
                self.env.execute({"command": FORCED_SUBMIT_COMMAND})
            except Submitted:
                raise
            raise LimitsExceeded(
                {
                    "role": "exit",
                    "content": "WallTimeExceeded",
                    "extra": {
                        "exit_status": "WallTimeExceeded",
                        "submission": "",
                    },
                }
            )

        def query(self) -> dict:
            if self._deadline_expired():
                self._submit_at_deadline()
            return super().query()

        def execute_actions(self, message: dict) -> list[dict]:
            actions = message.get("extra", {}).get("actions", [])
            is_submit = (
                len(actions) == 1
                and "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
                in str(actions[0].get("command", ""))
            )
            if self._deadline_expired() and not is_submit:
                self._submit_at_deadline()
            return super().execute_actions(message)

    return DeadlineAgent(model, env, **kwargs)


class DoubleBlindLogFilter(logging.Filter):
    """Redact local identity-bearing paths and Modal workspace names."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        message = re.sub(
            r"https://modal\.com/apps/[^/\s]+/",
            "https://modal.com/apps/<workspace>/",
            message,
        )
        message = re.sub(r"/home/[^/\s]+", "/home/<user>", message)
        message = re.sub(
            r"/work/[^/\s]+/[^/\s]+", "/work/<project>/<user>", message
        )
        record.msg = message
        record.args = ()
        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=3)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--selection-source", type=Path)
    parser.add_argument("--generation-seed", type=int, default=0x1352766)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--watermarking", choices=("none", "wllm"), default="none")
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--delta", type=float)
    parser.add_argument("--watermark-key", type=int, default=15485863)
    parser.add_argument("--z-threshold", type=float, default=4.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-context-tokens", type=int, default=200_000)
    parser.add_argument("--max-new-tokens", type=int, default=32_768)
    parser.add_argument("--max-generation-seconds", type=float, default=300.0)
    parser.add_argument("--agent-workers", type=int, default=1)
    parser.add_argument("--eval-workers", type=int, default=8)
    parser.add_argument("--modal-eval-timeout", type=int, default=1800)
    parser.add_argument("--swebench-python", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    args = parser.parse_args()

    if not 1 <= args.sample_size <= 500:
        parser.error("--sample-size must be in [1, 500]")
    if args.watermarking == "wllm":
        if args.gamma is None or not 0 < args.gamma < 1:
            parser.error("WLLM requires --gamma in (0, 1)")
        if args.delta is None or args.delta <= 0:
            parser.error("WLLM requires positive --delta")
    if args.max_new_tokens < 10_000:
        parser.error("--max-new-tokens must be at least 10000")
    if args.max_generation_seconds < 60:
        parser.error("--max-generation-seconds must be at least 60")
    if args.agent_workers <= 0 or args.eval_workers <= 0:
        parser.error("worker counts must be positive")
    if not args.swebench_python.is_file():
        parser.error(f"SWE-bench Python does not exist: {args.swebench_python}")
    if args.selection_source and not args.selection_source.is_file():
        parser.error(f"Selection source does not exist: {args.selection_source}")
    return args


def configure_logging(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)sZ %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    formatter.converter = time.gmtime
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    for handler in (logging.StreamHandler(), logging.FileHandler(data_dir / "run.log")):
        handler.setFormatter(formatter)
        handler.addFilter(DoubleBlindLogFilter())
        root.addHandler(handler)
    # SWE-ReX INFO messages include a workspace-specific Modal dashboard URL.
    # Keep experiment logs suitable for double-blind artifact sharing.
    logging.getLogger("rex-deploy").setLevel(logging.WARNING)
    logging.getLogger("rex_image_builder").setLevel(logging.WARNING)


def atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with WRITE_LOCK, path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def experiment_name(args: argparse.Namespace) -> str:
    if args.watermarking == "none":
        return "non-wm"
    return f"wllm-delta-{args.delta:g}-gamma-{args.gamma:g}-ngram-{NGRAM_LEN}"


def prediction_model_name(args: argparse.Namespace) -> str:
    return f"Qwen3.6-35B-A3B--mini-swe-agent-v2.2.6--{experiment_name(args)}"


def load_cases(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from datasets import load_dataset

    LOGGER.info("Loading Verified test split from the Hugging Face cache")
    dataset = load_dataset(VERIFIED_DATASET, split="test")
    if len(dataset) != 500:
        raise RuntimeError(f"Expected 500 Verified cases, found {len(dataset)}")
    by_id = {row["instance_id"]: dict(row) for row in dataset}

    if args.selection_source:
        source = json.loads(args.selection_source.read_text())
        source_ids = source.get("instance_ids")
        if not isinstance(source_ids, list):
            source_ids = [case["instance_id"] for case in source.get("cases", [])]
        if len(source_ids) < args.sample_size:
            raise RuntimeError("Selection source has fewer IDs than requested")
        selected_ids = source_ids[: args.sample_size]
        selection_method = "prefix_of_explicit_selection_source"
    else:
        selected_ids = random.Random(args.selection_seed).sample(
            sorted(by_id), args.sample_size
        )
        selection_method = "uniform_without_replacement"

    if len(set(selected_ids)) != args.sample_size:
        raise RuntimeError("Selected instance IDs are not unique")
    missing = [instance_id for instance_id in selected_ids if instance_id not in by_id]
    if missing:
        raise RuntimeError(f"Selection contains non-Verified IDs: {missing[:5]}")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset": VERIFIED_DATASET,
        "split": "test",
        "population": len(by_id),
        "sample_size": args.sample_size,
        "selection_seed": args.selection_seed,
        "selection_method": selection_method,
        "selection_source": str(args.selection_source) if args.selection_source else None,
        "instance_ids": selected_ids,
        "cases": [
            {
                "instance_id": instance_id,
                "repo": by_id[instance_id]["repo"],
                "difficulty": by_id[instance_id].get("difficulty"),
            }
            for instance_id in selected_ids
        ],
    }
    atomic_write_json(args.data_dir / "selection.json", manifest)
    return [by_id[instance_id] for instance_id in selected_ids], manifest


def load_official_config() -> dict[str, Any]:
    from minisweagent.config import builtin_config_dir, get_config_from_spec
    from minisweagent.utils.serialize import recursive_merge

    benchmark_dir = builtin_config_dir / "benchmarks"
    backticks = get_config_from_spec(str(benchmark_dir / "swebench_backticks.yaml"))
    modal = get_config_from_spec(str(benchmark_dir / "swebench_modal.yaml"))
    config = recursive_merge(backticks, modal)
    if config["agent"].get("step_limit") != 250:
        raise RuntimeError("Pinned official step_limit is not 250")
    if config["environment"].get("environment_class") != "swerex_modal":
        raise RuntimeError("Official Modal environment override was not applied")
    if config["environment"].get("timeout") != 60:
        raise RuntimeError("Pinned official command timeout is not 60 seconds")
    return config


def extract_added_code(patch: str) -> str:
    """Return all hunk-added lines, in diff order, without diff markers."""

    added: list[str] = []
    in_hunk = False
    for line in patch.splitlines(keepends=True):
        if line.startswith("diff --git "):
            in_hunk = False
            continue
        if line.startswith("@@"):
            in_hunk = True
            continue
        if in_hunk and line.startswith("+"):
            value = line[1:]
            added.append(value if value.endswith(("\n", "\r")) else value + "\n")
    return "".join(added)


def create_patch_detector(tokenizer, args: argparse.Namespace):
    if args.watermarking == "none":
        return None
    from src._sweet import WatermarkDetector

    return WatermarkDetector(
        vocab=list(tokenizer.get_vocab().values()),
        gamma=args.gamma,
        delta=args.delta,
        seeding_scheme="n_grams",
        ngram_len=NGRAM_LEN,
        hash_key=args.watermark_key,
        select_green_tokens=True,
        tokenizer=tokenizer,
        z_threshold=args.z_threshold,
    )


def detect_added_code(
    patch: str,
    detector,
    tokenizer,
    args: argparse.Namespace,
    *,
    serialize_detector: bool = True,
) -> tuple[str, dict[str, Any] | None]:
    added_code = extract_added_code(patch)
    if detector is None:
        return added_code, None
    import torch

    token_ids = tokenizer(added_code, add_special_tokens=False)["input_ids"]
    if not token_ids:
        return added_code, {
            "enabled": True,
            "scope": "all_patch_added_lines",
            "invalid": True,
            "reason": "no_added_code_tokens",
            "num_added_characters": len(added_code),
            "num_added_tokens": 0,
        }
    started = time.monotonic()
    detector_context = DETECT_LOCK if serialize_detector else nullcontext()
    with detector_context:
        raw = detector.detect(
            tokenized_text=torch.tensor(token_ids, dtype=torch.long),
            tokenized_prefix=torch.tensor([], dtype=torch.long),
            return_green_token_mask=False,
        )
    result: dict[str, Any] = {
        "enabled": True,
        "scope": "all_patch_added_lines",
        "invalid": bool(raw.get("invalid", False)),
        "num_added_characters": len(added_code),
        "num_added_tokens": len(token_ids),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "z_threshold": args.z_threshold,
    }
    for key in ("num_tokens_scored", "num_green_tokens"):
        result[key] = int(raw[key]) if raw.get(key) is not None else None
    for key in ("green_fraction", "z_score", "p_value", "confidence"):
        result[key] = float(raw[key]) if raw.get(key) is not None else None
    result["prediction"] = bool(raw.get("prediction", False))
    return added_code, result


def run_agent_case(
    *,
    case: dict[str, Any],
    index: int,
    total: int,
    data_dir: Path,
    config: dict[str, Any],
    model: QwenHFTextModel,
    detector,
    args: argparse.Namespace,
) -> dict[str, Any]:
    from minisweagent.run.benchmarks.swebench import get_sb_environment

    instance_id = case["instance_id"]
    case_dir = data_dir / "trajectories" / f"case-{index:03d}-{instance_id}"
    case_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = case_dir / f"{instance_id}.traj.json"
    LOGGER.info("[%d/%d] Starting mini-swe-agent for %s", index, total, instance_id)
    started = time.monotonic()
    env = None
    agent = None
    info: dict[str, Any] = {}
    error: str | None = None
    try:
        env = get_sb_environment(config, case)
        agent_config = {
            **config["agent"],
            "output_path": trajectory_path,
            # The official Modal sandbox lifetime is 1800 seconds. Leave time
            # for an in-flight model turn and a best-effort final submission.
            "wall_time_limit_seconds": AGENT_WALL_TIME_LIMIT_SECONDS,
        }
        agent = create_deadline_agent(model, env, **agent_config)
        info = agent.run(case["problem_statement"])
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        LOGGER.exception("[%d/%d] Agent failed for %s", index, total, instance_id)
        info = {"exit_status": type(exc).__name__, "submission": ""}
    finally:
        if agent is not None:
            agent.save(
                trajectory_path,
                {
                    "instance_id": instance_id,
                    "info": {"runner_error": error},
                },
            )
        if env is not None:
            env.stop()

    patch = info.get("submission") or ""
    added_code, detection = detect_added_code(
        patch, detector, model.tokenizer, args
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
    z_text = (
        f" z={detection.get('z_score')}" if isinstance(detection, dict) else ""
    )
    LOGGER.info(
        "[%d/%d] Agent result %s: exit=%s patch_chars=%d added_chars=%d%s",
        index,
        total,
        instance_id,
        row["exit_status"],
        len(patch),
        len(added_code),
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


def summarize(
    *,
    data_dir: Path,
    args: argparse.Namespace,
    selected_ids: list[str],
    rows: dict[str, dict[str, Any]],
    official_report: dict[str, Any] | None,
) -> dict[str, Any]:
    valid_detections = [
        row["detection"]
        for row in rows.values()
        if isinstance(row.get("detection"), dict)
        and not row["detection"].get("invalid")
        and row["detection"].get("z_score") is not None
    ]
    z_scores = [float(item["z_score"]) for item in valid_detections]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "run_id": args.run_id,
        "experiment": experiment_name(args),
        "selected_cases": len(selected_ids),
        "completed_agent_cases": len(rows),
        "nonempty_patches": sum(bool(row["model_patch"].strip()) for row in rows.values()),
        "submitted_agent_cases": sum(
            row.get("exit_status") == "Submitted" for row in rows.values()
        ),
        "resolved_cases": (
            official_report.get("resolved_instances") if official_report else None
        ),
        "solve_rate": (
            official_report.get("resolved_instances", 0) / len(selected_ids)
            if official_report and selected_ids
            else None
        ),
        "watermarking": args.watermarking,
        "delta": args.delta,
        "gamma": args.gamma,
        "ngram_len": NGRAM_LEN,
        "detection_scope": "all_patch_added_lines" if args.watermarking == "wllm" else None,
        "valid_detection_cases": len(valid_detections),
        "mean_z_score": statistics.fmean(z_scores) if z_scores else None,
        "median_z_score": statistics.median(z_scores) if z_scores else None,
        "min_z_score": min(z_scores) if z_scores else None,
        "max_z_score": max(z_scores) if z_scores else None,
        "auroc_vs_standard_normal": ideal_auroc(z_scores),
    }
    atomic_write_json(data_dir / "summary.json", summary)
    LOGGER.info("Final experiment summary: %s", json.dumps(summary, sort_keys=True))
    return summary


def main() -> None:
    args = parse_args()
    configure_logging(args.data_dir)
    mini_version = importlib.metadata.version("mini-swe-agent")
    swerex_version = importlib.metadata.version("swe-rex")
    if mini_version != "2.2.6":
        raise RuntimeError(f"Expected mini-swe-agent 2.2.6, found {mini_version}")
    LOGGER.info("Using mini-swe-agent=%s swe-rex=%s", mini_version, swerex_version)

    cases, manifest = load_cases(args)
    config = load_official_config()
    model_kwargs = {
        key: value
        for key, value in config.get("model", {}).items()
        if key in {"observation_template", "format_error_template", "multimodal_regex"}
    }
    model = QwenHFTextModel(
        model_name=args.model_id,
        watermarking=args.watermarking,
        gamma=args.gamma,
        delta=args.delta,
        ngram_len=NGRAM_LEN,
        watermark_key=args.watermark_key,
        generation_seed=args.generation_seed,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_context_tokens=args.max_context_tokens,
        max_new_tokens=args.max_new_tokens,
        max_generation_seconds=args.max_generation_seconds,
        local_files_only=not args.allow_model_download,
        **model_kwargs,
    )
    detector = create_patch_detector(model.tokenizer, args)

    settings = {
        "schema_version": SCHEMA_VERSION,
        "model_id": args.model_id,
        "mini_swe_agent_version": mini_version,
        "swe_rex_version": swerex_version,
        "official_configs": ["swebench_backticks.yaml", "swebench_modal.yaml"],
        "step_limit": config["agent"]["step_limit"],
        "agent_wall_time_limit_seconds": AGENT_WALL_TIME_LIMIT_SECONDS,
        "command_timeout": config["environment"]["timeout"],
        "modal_runtime_timeout": config["environment"]["runtime_timeout"],
        "modal_sandbox_kwargs": config["environment"]["modal_sandbox_kwargs"],
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_context_tokens": args.max_context_tokens,
        "max_new_tokens": args.max_new_tokens,
        "max_generation_seconds": args.max_generation_seconds,
        "watermarking": args.watermarking,
        "gamma": args.gamma,
        "delta": args.delta,
        "ngram_len": NGRAM_LEN,
        "watermark_key": args.watermark_key,
        "detector_scope": "all_patch_added_lines",
        "selection": manifest,
    }
    atomic_write_json(args.data_dir / "settings.json", settings)

    rows: dict[str, dict[str, Any]] = {}
    eval_futures: dict[Future, str] = {}
    heartbeat = GPUHeartbeat(args.data_dir, interval=15.0, busy_seconds=1.0)
    heartbeat.start()
    try:
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
                        args=args,
                    ): (index, case["instance_id"])
                    for index, case in enumerate(cases, 1)
                }
                for future in as_completed(agent_futures):
                    index, instance_id = agent_futures[future]
                    try:
                        row = future.result()
                    except Exception as exc:
                        LOGGER.error(
                            "Uncaught case worker error for %s: %s\n%s",
                            instance_id,
                            exc,
                            traceback.format_exc(),
                        )
                        row = {
                            "instance_id": instance_id,
                            "model_name_or_path": prediction_model_name(args),
                            "model_patch": "",
                            "status": "uncaught_error",
                            "runner_error": f"{type(exc).__name__}: {exc}",
                        }
                    rows[instance_id] = row
                    write_predictions(
                        args.data_dir / "predictions.jsonl",
                        manifest["instance_ids"],
                        rows,
                    )
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
    finally:
        heartbeat.stop()
    if heartbeat.error:
        raise RuntimeError(f"GPU heartbeat failed: {heartbeat.error}")

    official_report = None
    if not args.skip_evaluation:
        official_report = aggregate_official_reports(
            data_dir=args.data_dir,
            run_id=args.run_id,
            selected_ids=manifest["instance_ids"],
            prediction_name=prediction_model_name(args),
        )
    summarize(
        data_dir=args.data_dir,
        args=args,
        selected_ids=manifest["instance_ids"],
        rows=rows,
        official_report=official_report,
    )


if __name__ == "__main__":
    main()
