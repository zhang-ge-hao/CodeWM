#!/usr/bin/env python3
"""Detect WLLM on saved method-scoped patches without rerunning generation."""

from __future__ import annotations

import argparse
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from run_experiment import atomic_write_json, atomic_write_jsonl, configure_logging
from run_mini_experiment import create_patch_detector, detect_added_code
from run_python_minifier_postprocess import make_summary


LOGGER = logging.getLogger("swe_bench_experiment")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--experiment", action="append", default=[])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")
    return args


def detector_args(settings: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        watermarking=settings["watermarking"],
        gamma=settings.get("gamma"),
        delta=settings.get("delta"),
        watermark_key=settings["watermark_key"],
        z_threshold=4.0,
    )


def update_experiment(
    experiment_dir: Path,
    *,
    tokenizer,
    workers: int,
) -> dict[str, Any]:
    settings_path = experiment_dir / "settings.json"
    settings = json.loads(settings_path.read_text())
    selection = json.loads((experiment_dir / "selection.json").read_text())
    selected_ids = list(selection["instance_ids"])
    result_paths = sorted((experiment_dir / "cases").glob("case-*/result.json"))
    rows = {
        row["instance_id"]: row
        for path in result_paths
        for row in [json.loads(path.read_text())]
    }
    missing = set(selected_ids) - set(rows)
    if missing:
        raise RuntimeError(f"Missing case results in {experiment_dir}: {sorted(missing)}")

    args = detector_args(settings)
    thread_state = threading.local()

    def score_one(result_path: Path) -> dict[str, Any]:
        row = json.loads(result_path.read_text())
        if row["obfuscation"]["status"] != "transformed":
            return row
        existing = row.get("post_detection")
        if isinstance(existing, dict) and existing.get("enabled"):
            return row
        detector = getattr(thread_state, "detector", None)
        if detector is None:
            detector = create_patch_detector(tokenizer, args)
            thread_state.detector = detector
        patch = (result_path.parent / "obfuscated_patch.diff").read_text()
        started = time.monotonic()
        added_code, detection = detect_added_code(
            patch,
            detector,
            tokenizer,
            args,
            serialize_detector=False,
        )
        row["post_detection"] = detection
        row["detection"] = detection
        (result_path.parent / "added_code_post.txt").write_text(added_code)
        atomic_write_json(result_path, row)
        LOGGER.info(
            "%s %s: tokens=%s z=%s elapsed=%.2fs",
            experiment_dir.name,
            row["instance_id"],
            detection.get("num_added_tokens") if detection else None,
            detection.get("z_score") if detection else None,
            time.monotonic() - started,
        )
        return row

    if settings["watermarking"] == "wllm":
        transformed_paths = [
            path
            for path in result_paths
            if rows[json.loads(path.read_text())["instance_id"]]["obfuscation"][
                "status"
            ]
            == "transformed"
        ]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(score_one, path): path for path in transformed_paths}
            for future in as_completed(futures):
                row = future.result()
                rows[row["instance_id"]] = row

    settings["detection_skipped"] = False
    settings["detection_scope"] = (
        "all_final_patch_added_lines"
        if settings["watermarking"] == "wllm"
        else None
    )
    atomic_write_json(settings_path, settings)
    atomic_write_jsonl(
        experiment_dir / "case_results.jsonl", (rows[value] for value in selected_ids)
    )
    original_report = json.loads(
        (Path(settings["source_dir"]) / "aggregate_official_report.json").read_text()
    )
    summary = make_summary(
        rows=rows,
        selected_ids=selected_ids,
        original_report=original_report,
        post_report=None,
        settings=settings,
    )
    atomic_write_json(experiment_dir / "summary.json", summary)
    LOGGER.info("Completed %s: %s", experiment_dir.name, json.dumps(summary))
    return summary


def main() -> None:
    args = parse_args()
    configure_logging(args.root)
    names = args.experiment or sorted(
        path.name
        for path in args.root.iterdir()
        if path.is_dir() and (path / "settings.json").is_file()
    )
    experiment_dirs = [args.root / name for name in names]
    wllm_dirs = [
        path
        for path in experiment_dirs
        if json.loads((path / "settings.json").read_text())["watermarking"] == "wllm"
    ]
    tokenizer = None
    if wllm_dirs:
        from transformers import AutoTokenizer

        settings = json.loads((wllm_dirs[0] / "settings.json").read_text())
        tokenizer = AutoTokenizer.from_pretrained(
            settings["source_settings"]["model_id"], local_files_only=True
        )
    for experiment_dir in experiment_dirs:
        update_experiment(experiment_dir, tokenizer=tokenizer, workers=args.workers)


if __name__ == "__main__":
    main()
