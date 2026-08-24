#!/usr/bin/env python3
"""Run the vendored official ACW-s pipeline with minimal compatibility fixes."""

from __future__ import annotations

import importlib.machinery
import json
import os
from pathlib import Path
import sys
import types


ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT / "upstream"


def empty_detection_result(
    pass_info: dict,
    pass_results: object,
) -> dict:
    """Represent a prompt-only generation without dropping its benchmark task.

    The official evaluator returns ``False`` when none of a task's completions
    contains a token after the prompt. Its outer loop then silently removes the
    whole task from Pass@k and ROC aggregation. For an all-task analysis, a
    prompt-only completion is a functional failure with no watermark evidence,
    i.e. z=0 and prediction=False.
    """

    return {
        "pass@1": pass_info["pass@1"],
        "pass@10": pass_info.get("pass@10", 0),
        "pass_results": pass_results,
        "entropy": [],
        "len": 0,
        "num_tokens_generated": 0,
        "num_tokens_scored": 0,
        "num_green_tokens": 0,
        "watermarking_fraction": 0.0,
        "green_fraction": 0.0,
        "z_score": 0.0,
        "p_value": 1.0,
        "prediction": False,
        "empty_generation": True,
    }


def install_keep_empty_evaluator(official_main) -> None:
    """Patch only the upstream task-dropping policy, not ACW-s scoring."""

    original_evaluate_code = official_main.evaluate_code

    def evaluate_code_keep_empty(*args, **kwargs):
        task = kwargs.get("task")
        if task is None:
            # Upstream itself always passes task by keyword. Keep positional
            # compatibility for direct callers and tests.
            task = args[4]

        original_process_results = task.process_results
        captured: dict[str, object] = {}

        def capture_process_results(generations, references):
            value = original_process_results(generations, references)
            captured["value"] = value
            return value

        task.process_results = capture_process_results
        try:
            result = original_evaluate_code(*args, **kwargs)
        finally:
            task.process_results = original_process_results

        if result is not False:
            return result
        if "value" not in captured:
            raise RuntimeError(
                "upstream evaluate_code returned False before Pass@k was computed"
            )
        pass_info, pass_results = captured["value"]
        return empty_detection_result(pass_info, pass_results)

    official_main.evaluate_code = evaluate_code_keep_empty


def _install_noop_debug_modules() -> None:
    """Supply only dependencies used for logging/debugging by upstream code."""

    wandb = types.ModuleType("wandb")
    wandb.__spec__ = importlib.machinery.ModuleSpec("wandb", loader=None)
    wandb.init = lambda *args, **kwargs: None
    wandb.log = lambda *args, **kwargs: None
    wandb.save = lambda *args, **kwargs: None
    wandb.finish = lambda *args, **kwargs: None
    sys.modules["wandb"] = wandb

    ipdb = types.ModuleType("ipdb")
    ipdb.__spec__ = importlib.machinery.ModuleSpec("ipdb", loader=None)

    def fail_instead_of_debugging(*args, **kwargs):
        raise RuntimeError("The upstream code attempted to enter ipdb")

    ipdb.set_trace = fail_instead_of_debugging
    sys.modules["ipdb"] = ipdb


def main() -> None:
    # Let Accelerate inspect the real environment before the no-op wandb module
    # is registered.  This prevents it from treating the stub as a tracker.
    import accelerate  # noqa: F401

    _install_noop_debug_modules()
    sys.path.insert(0, str(UPSTREAM))
    os.chdir(UPSTREAM)

    from src import main as official_main
    from src import utils as official_utils

    def original_argument(flag: str) -> str | None:
        try:
            return sys.argv[sys.argv.index(flag) + 1]
        except (ValueError, IndexError):
            return None

    original_model = original_argument("--model")
    original_output_dir = original_argument("--output_dir")
    args = official_main.parse_args()
    # parse_args() lower-cases every string, including case-sensitive model IDs
    # and Linux paths. Restore these two values exactly as supplied.
    if original_model is not None:
        args.model = original_model
    if original_output_dir is not None:
        args.output_dir = original_output_dir
    if args.wm != "sweetcode":
        raise ValueError("This reproduction wrapper only permits --wm sweetcode")

    skip_marker = Path(args.output_dir).parent / f"SKIP_{args.task_name.upper()}"
    if skip_marker.exists():
        print(f"Skipping {args.task_name}: marker present at {skip_marker}")
        return

    # The checkpoint and datasets are public.  The upstream tokenizer helper
    # hard-codes token=True, which fails on a clean Unity account with no HF
    # login.  This is an access fix only; it does not change prompts or tokens.
    args.use_auth_token = False

    def get_public_tokenizer(model_name: str):
        return official_utils.get_tokenizer(model_name, use_auth_token=False)

    official_main.get_tokenizer = get_public_tokenizer
    install_keep_empty_evaluator(official_main)

    run_metadata = {
        "upstream_repository": "https://github.com/TimeLovercc/code-watermark",
        "upstream_commit": "4291a8e07abda0bc20560626f98340a90059be67",
        "arguments": vars(args),
        "compatibility_changes": [
            "disabled wandb logging",
            "disabled interactive ipdb",
            "used anonymous access for public Hugging Face artifacts",
            "skipped upstream's optional 13 GiB allocator warm-up",
            "preserved case in the model ID and output path",
            "retained prompt-only tasks as functional failures with z_score=0",
        ],
    }
    metadata_path = Path(args.output_dir) / "run-metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(run_metadata, indent=2, default=str))

    official_main.main(args)


if __name__ == "__main__":
    main()
