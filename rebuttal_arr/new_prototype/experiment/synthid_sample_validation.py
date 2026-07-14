"""Sample every saved SynthID config and reproduce original/obfuscated scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from transformers import SynthIDTextWatermarkLogitsProcessor

from .common import RESULT_ROOT, iter_jsonl, run_root
from .detectors import _cached_snapshot, load_tokenizer
from .synthid_forensics import LEGACY_FIXED_KEYS, tensor_digest, token_digest


def synthid_directories() -> list[Path]:
    directories = sorted(
        path
        for path in RESULT_ROOT.glob("*--synthid--*/*")
        if path.is_dir() and path.name.isdigit()
    )
    if len(directories) != 40:
        raise RuntimeError(f"expected 40 SynthID config directories, found {len(directories)}")
    return directories


def deterministic_sample(
    rows: Iterable[Mapping[str, Any]], *, config_key: str, count: int
) -> list[dict[str, Any]]:
    ranked = sorted(
        (dict(row) for row in rows),
        key=lambda row: hashlib.sha256(
            f"{config_key}\0{row['id']}".encode("utf-8")
        ).digest(),
    )
    if len(ranked) < count:
        raise RuntimeError(f"{config_key} has only {len(ranked)} generation records")
    return ranked[:count]


def score_text(
    processor: SynthIDTextWatermarkLogitsProcessor,
    tokenizer: Any,
    text: str,
    *,
    device: str,
    ngram_len: int,
) -> dict[str, Any]:
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    if input_ids.size(-1) < ngram_len:
        return {
            "num_input_tokens": int(input_ids.numel()),
            "input_ids_sha256": token_digest(input_ids),
            "g_values_shape": [int(input_ids.size(0)), 0, len(LEGACY_FIXED_KEYS)],
            "g_values_sha256": hashlib.sha256(b"").hexdigest(),
            "num_g_values": 0,
            "num_ones": 0,
            "z_score": 0.0,
            "p_value": 0.5,
        }
    g_values = processor.compute_g_values(input_ids.to(device))
    flattened = g_values.reshape(-1)
    count = int(flattened.numel())
    ones = int(flattened.to(torch.int64).sum().item())
    z_score = (
        (ones - count * 0.5) / math.sqrt(count * 0.25)
        if count
        else 0.0
    )
    return {
        "num_input_tokens": int(input_ids.numel()),
        "input_ids_sha256": token_digest(input_ids),
        "g_values_shape": list(g_values.shape),
        "g_values_sha256": tensor_digest(flattened),
        "num_g_values": count,
        "num_ones": ones,
        "z_score": z_score,
        "p_value": 0.5 * math.erfc(z_score / math.sqrt(2.0)),
    }


def compare_score(saved: Mapping[str, Any], recomputed: Mapping[str, Any]) -> dict[str, Any]:
    saved_z = float(saved["z_score"])
    saved_p = float(saved["p_value"])
    z_error = abs(saved_z - float(recomputed["z_score"]))
    p_error = abs(saved_p - float(recomputed["p_value"]))
    return {
        **dict(recomputed),
        "saved_z_score": saved_z,
        "saved_p_value": saved_p,
        "z_absolute_error": z_error,
        "p_absolute_error": p_error,
        "z_exact_match": math.isclose(saved_z, float(recomputed["z_score"]), rel_tol=0.0, abs_tol=1e-12),
        "p_exact_match": math.isclose(saved_p, float(recomputed["p_value"]), rel_tol=0.0, abs_tol=1e-12),
    }


def validate_config(directory: Path, *, sample_count: int, device: str) -> dict[str, Any]:
    generations = list(iter_jsonl(directory / "generate.jsonl"))
    obfuscations = list(iter_jsonl(directory / "obfuscate.jsonl"))
    folder_parts = directory.parent.name.split("--")
    config_key = f"{directory.parent.name}--{directory.name}"

    obfuscations_by_generation: dict[str, list[dict[str, Any]]] = {}
    for row in obfuscations:
        obfuscations_by_generation.setdefault(str(row["gen_task_id"]), []).append(row)

    def is_scored(row: Mapping[str, Any]) -> bool:
        return all(row.get(field) is not None for field in ("g4d", "z_score", "p_value"))

    eligible_generations = []
    for generation in generations:
        related = obfuscations_by_generation.get(str(generation["id"]), [])
        if is_scored(generation) and len(related) == 2 and all(is_scored(row) for row in related):
            eligible_generations.append(generation)
    sampled = deterministic_sample(
        eligible_generations,
        config_key=config_key,
        count=sample_count,
    )

    model_name = str(generations[0]["model_name"])
    tokenizer = load_tokenizer(model_name)
    snapshot = _cached_snapshot(model_name)
    cases: list[dict[str, Any]] = []
    for generation in sampled:
        related = sorted(
            obfuscations_by_generation.get(str(generation["id"]), []),
            key=lambda row: str(row["obf_name"]),
        )
        if len(related) != 2:
            raise RuntimeError(
                f"{config_key}/{generation['task_name']} has {len(related)} obfuscations"
            )
        seed = int(generation["custom_seed"])
        ngram_len = int(generation["ngram_len"])
        processor = SynthIDTextWatermarkLogitsProcessor(
            device=device,
            ngram_len=ngram_len,
            keys=list(LEGACY_FIXED_KEYS),
            sampling_table_size=2**16,
            sampling_table_seed=seed,
            context_history_size=1024,
        )
        records = [("Original", generation)] + [
            (str(row["obf_name"]), row) for row in related
        ]
        comparisons = {}
        for label, record in records:
            recomputed = score_text(
                processor,
                tokenizer,
                str(record["g4d"]),
                device=device,
                ngram_len=ngram_len,
            )
            comparisons[label] = compare_score(record, recomputed)
        cases.append(
            {
                "task_name": generation["task_name"],
                "generation_id": generation["id"],
                "custom_seed": seed,
                "ngram_len": ngram_len,
                "comparisons": comparisons,
            }
        )

    comparisons = [
        comparison
        for case in cases
        for comparison in case["comparisons"].values()
    ]
    return {
        "config_key": config_key,
        "model_slug": folder_parts[0],
        "dataset": folder_parts[2],
        "config_id": directory.name,
        "model_name": model_name,
        "temperature": generations[0]["temperature"],
        "tokenizer_snapshot": None if snapshot is None else str(snapshot),
        "generation_count": len(generations),
        "obfuscation_count": len(obfuscations),
        "eligible_common_cohort_count": len(eligible_generations),
        "sample_count": sample_count,
        "score_comparison_count": len(comparisons),
        "z_match_count": sum(result["z_exact_match"] for result in comparisons),
        "p_match_count": sum(result["p_exact_match"] for result in comparisons),
        "max_z_absolute_error": max(result["z_absolute_error"] for result in comparisons),
        "max_p_absolute_error": max(result["p_absolute_error"] for result in comparisons),
        "cases": cases,
    }


def build_report(*, sample_count: int, device: str) -> dict[str, Any]:
    configs = [
        validate_config(directory, sample_count=sample_count, device=device)
        for directory in synthid_directories()
    ]
    comparison_count = sum(config["score_comparison_count"] for config in configs)
    z_matches = sum(config["z_match_count"] for config in configs)
    p_matches = sum(config["p_match_count"] for config in configs)
    return {
        "schema_version": 1,
        "detector_contract": {
            "seed": "saved custom_seed",
            "keys": list(LEGACY_FIXED_KEYS),
            "processor": "transformers.SynthIDTextWatermarkLogitsProcessor",
            "device": device,
            "sampling_table_size": 2**16,
            "context_history_size": 1024,
        },
        "summary": {
            "config_count": len(configs),
            "samples_per_config": sample_count,
            "original_comparison_count": len(configs) * sample_count,
            "obfuscation_comparison_count": len(configs) * sample_count * 2,
            "score_comparison_count": comparison_count,
            "z_match_count": z_matches,
            "p_match_count": p_matches,
            "all_z_scores_match": z_matches == comparison_count,
            "all_p_values_match": p_matches == comparison_count,
            "max_z_absolute_error": max(config["max_z_absolute_error"] for config in configs),
            "max_p_absolute_error": max(config["max_p_absolute_error"] for config in configs),
        },
        "configs": configs,
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="rw100-useful-v1")
    parser.add_argument("--samples-per-config", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.samples_per_config < 1:
        parser.error("--samples-per-config must be positive")
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    output = (
        run_root(args.run_id)
        / "forensics"
        / f"synthid_sample_validation_{args.samples_per_config}.json"
    )
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {output}; pass --overwrite")
    report = build_report(sample_count=args.samples_per_config, device=args.device)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(output)
    print(json.dumps(report["summary"], indent=2, sort_keys=True), flush=True)
    print(f"report={output}", flush=True)
    return 0 if report["summary"]["all_z_scores_match"] and report["summary"]["all_p_values_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
