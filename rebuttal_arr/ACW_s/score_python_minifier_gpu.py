#!/usr/bin/env python3
"""Score Python-Minifier outputs with ACW-s on the official retained cohort."""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
HUMANEVAL_PATH = REPO_ROOT / "data" / "original" / "humaneval-x_py.jsonl"
EXPECTED_RETAINED_TASKS = 160


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def benchmark_rows() -> list[dict[str, Any]]:
    rows = read_jsonl(HUMANEVAL_PATH)
    if len(rows) != 164:
        raise ValueError(f"expected 164 HumanEval rows, found {len(rows)}")
    return rows


def entry_point(prompt: str) -> str:
    tree = ast.parse(prompt)
    functions = [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if not functions:
        raise ValueError("HumanEval prompt has no top-level function")
    return functions[-1]


def attacked_suffix(code: str, function_name: str) -> str:
    """Use the same Python-Minifier suffix extraction as the paper pipeline."""

    lines = code.split("\n")
    for index, line in enumerate(lines):
        if function_name not in line:
            continue
        inline = "):".join(line.split("):")[1:])
        if inline:
            inline = f"    {inline.strip()}\n"
        return inline + "\n".join(lines[index + 1 :])
    return ""


def load_attack_rows(cpu_root: Path) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for path in sorted((cpu_root / "shards").glob("part-*.jsonl")):
        for row in read_jsonl(path):
            if row.get("reconstructed_empty_task"):
                continue
            grouped.setdefault(int(row["task_index"]), []).append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["completion_index"]))
    return grouped


def retained_task_indices(source_results: Path) -> list[int]:
    indices = sorted(
        int(path.stem.removeprefix("sample_"))
        for path in source_results.glob("sample_*.json")
    )
    if len(indices) != EXPECTED_RETAINED_TASKS:
        raise ValueError(
            f"expected {EXPECTED_RETAINED_TASKS} retained samples, found {len(indices)}"
        )
    return indices


def shard_bounds(size: int, shard_index: int, shard_count: int) -> tuple[int, int]:
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("invalid shard index/count")
    width = math.ceil(size / shard_count)
    return shard_index * width, min(size, (shard_index + 1) * width)


def score_sequence(model, tokenizer, detector, prompt: str, suffix: str, max_length: int):
    import torch

    prefix_ids = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).input_ids.squeeze(0).to(model.device)
    token_ids = tokenizer(
        prompt + suffix,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).input_ids.squeeze(0).to(model.device)
    prefix_len = len(prefix_ids)
    if len(token_ids) <= prefix_len:
        return None
    if not torch.equal(token_ids[: prefix_len - 1], prefix_ids[:-1]):
        raise ValueError("attacked sequence no longer begins with the oracle prompt")

    with torch.inference_mode():
        logits = model(input_ids=token_ids.unsqueeze(0), return_dict=True).logits[0]
        probabilities = torch.softmax(logits, dim=-1)
        entropy = -torch.where(
            probabilities > 0,
            probabilities * probabilities.log(),
            probabilities.new_tensor(0.0),
        ).sum(dim=-1)
        shifted_entropy = torch.cat(
            [entropy.new_zeros(1), entropy[:-1]], dim=0
        )
        shifted_scores = torch.cat(
            [logits.new_zeros((1, logits.shape[1])), logits[:-1]], dim=0
        )
        result = detector.detect(
            token_ids=token_ids,
            prefix_len=prefix_len,
            entropies=shifted_entropy,
            source_scores=shifted_scores,
        )
    if result.pop("invalid", False):
        return None
    result.pop("green_token_mask", None)
    result["len"] = len(token_ids) - prefix_len
    return result


def run_shard(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import sys

    sys.path.insert(0, str(HERE))
    from acw_s import ACWSDetector

    shard_index = args.shard_index
    if shard_index is None:
        value = os.environ.get("SLURM_ARRAY_TASK_ID")
        if value is None:
            raise ValueError("--shard-index or SLURM_ARRAY_TASK_ID is required")
        shard_index = int(value)

    indices = retained_task_indices(args.source_results)
    start, end = shard_bounds(len(indices), shard_index, args.shard_count)
    selected_indices = indices[start:end]
    attacks = load_attack_rows(args.cpu_root)
    benchmarks = benchmark_rows()
    if set(attacks) != set(indices):
        raise ValueError("CPU attack cohort differs from retained baseline cohort")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        token=False,
        local_files_only=args.local_files_only,
        truncation_side="left",
        padding_side="right",
    )
    if not tokenizer.eos_token:
        tokenizer.eos_token = tokenizer.bos_token
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        token=False,
        local_files_only=args.local_files_only,
    ).to("cuda")
    if len(tokenizer) != model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))
    model.eval()
    detector = ACWSDetector(
        vocab_size=len(tokenizer),
        gamma=0.5,
        delta=2.0,
        entropy_threshold=1.2,
        z_threshold=4.0,
    )

    score_dir = args.output_root / "scores"
    score_dir.mkdir(parents=True, exist_ok=True)
    for position, task_index in enumerate(selected_indices, start=1):
        output_path = score_dir / f"task_{task_index}.json"
        if output_path.is_file() and not args.overwrite:
            print(
                f"shard={shard_index} task={task_index} "
                f"progress={position}/{len(selected_indices)} cached",
                flush=True,
            )
            continue
        prompt = str(benchmarks[task_index]["prompt"]).strip()
        function_name = entry_point(prompt)
        attacked_result = None
        selected_completion = None
        skipped_transform_failures: list[int] = []
        skipped_empty_suffixes: list[int] = []
        for row in attacks[task_index]:
            completion_index = int(row["completion_index"])
            transformed = row["python_minifier"]
            if not transformed["ok"]:
                skipped_transform_failures.append(completion_index)
                continue
            suffix = attacked_suffix(str(transformed["code"]), function_name)
            if not suffix:
                skipped_empty_suffixes.append(completion_index)
                continue
            attacked_result = score_sequence(
                model,
                tokenizer,
                detector,
                prompt,
                suffix,
                args.max_length,
            )
            if attacked_result is not None:
                selected_completion = completion_index
                break

        if attacked_result is None:
            attacked_result = {
                "num_tokens_generated": 0,
                "num_tokens_scored": 0,
                "num_green_tokens": 0,
                "watermarking_fraction": 0.0,
                "green_fraction": 0.0,
                "z_score": 0.0,
                "p_value": 1.0,
                "prediction": False,
                "len": 0,
                "invalid_attack_output": True,
            }

        baseline = read_json(args.source_results / f"sample_{task_index}.json")
        record = {
            "task_index": task_index,
            "source_task_id": benchmarks[task_index]["task_id"],
            "selected_completion_index": selected_completion,
            "skipped_transform_failure_indices": skipped_transform_failures,
            "skipped_empty_suffix_indices": skipped_empty_suffixes,
            "attacked_metrics": attacked_result,
            "original_machine_z_score": baseline["metrics"]["z_score"],
            "human_z_score": baseline["human_metrics"]["z_score"],
        }
        write_json_atomic(output_path, record)
        print(
            f"shard={shard_index} task={task_index} "
            f"completion={selected_completion} z={attacked_result['z_score']:.6f} "
            f"progress={position}/{len(selected_indices)}",
            flush=True,
        )


def roc_summary(negative: list[float], positive: list[float]) -> dict[str, float]:
    from sklearn import metrics
    import numpy as np

    scores = np.concatenate([np.asarray(negative), np.asarray(positive)])
    labels = np.concatenate([np.zeros(len(negative)), np.ones(len(positive))])
    fpr, tpr, _ = metrics.roc_curve(labels, scores, pos_label=1)

    def value_at(limit: float) -> float:
        values = [float(t) for f, t in zip(fpr, tpr) if float(f) <= limit]
        if not values:
            raise ValueError(f"no ROC point at FPR <= {limit}")
        return values[-1]

    return {
        "roc_auc": float(metrics.auc(fpr, tpr)),
        "tpr_at_fpr_0": value_at(0.0),
        "tpr_at_fpr_01": value_at(0.01),
        "tpr_at_fpr_05": value_at(0.05),
    }


def aggregate(args: argparse.Namespace) -> None:
    indices = retained_task_indices(args.source_results)
    records = []
    for task_index in indices:
        path = args.output_root / "scores" / f"task_{task_index}.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing attacked score: {path}")
        records.append(read_json(path))

    human = [float(row["human_z_score"]) for row in records]
    original = [float(row["original_machine_z_score"]) for row in records]
    attacked = [float(row["attacked_metrics"]["z_score"]) for row in records]
    original_detection = roc_summary(human, original)
    attacked_detection = roc_summary(human, attacked)
    cpu_metrics = read_json(args.cpu_root / "metrics-official160.json")
    result = {
        "dataset": "HumanEval",
        "cohort": "official_retained_160",
        "task_count": len(records),
        "excluded_task_indices": [122, 137, 144, 149],
        "python_minifier": {
            "pass_at_1": cpu_metrics["attacked_pass_at_1"],
            "pass_at_10": cpu_metrics["attacked_pass_at_10"],
            "transform_success_count": cpu_metrics["python_minifier_success_count"],
        },
        "original_detection_recomputed": original_detection,
        "attacked_detection": attacked_detection,
        "detection_change": {
            key: attacked_detection[key] - original_detection[key]
            for key in attacked_detection
        },
        "invalid_attack_task_count": sum(
            bool(row["attacked_metrics"].get("invalid_attack_output"))
            for row in records
        ),
        "selected_nonzero_completion_count": sum(
            row["selected_completion_index"] not in (None, 0) for row in records
        ),
    }
    write_json_atomic(args.output_root / "metrics.json", result)
    print(json.dumps(result, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    score = subparsers.add_parser("score-shard")
    score.add_argument("--source-results", type=Path, required=True)
    score.add_argument("--cpu-root", type=Path, required=True)
    score.add_argument("--output-root", type=Path, required=True)
    # Match the cache key created by the completed upstream reproduction. HF
    # repository lookup is case-insensitive online, but offline cache lookup is
    # not and the upstream parser lower-cased this ID.
    score.add_argument("--model", default="infly/opencoder-1.5b-instruct")
    score.add_argument("--max-length", type=int, default=2048)
    score.add_argument("--shard-index", type=int)
    score.add_argument("--shard-count", type=int, default=4)
    score.add_argument("--local-files-only", action="store_true")
    score.add_argument("--overwrite", action="store_true")
    score.set_defaults(function=run_shard)

    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--source-results", type=Path, required=True)
    aggregate_parser.add_argument("--cpu-root", type=Path, required=True)
    aggregate_parser.add_argument("--output-root", type=Path, required=True)
    aggregate_parser.set_defaults(function=aggregate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
