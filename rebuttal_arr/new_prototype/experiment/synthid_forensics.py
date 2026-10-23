"""Explain and reproduce saved SynthID scores under historical configurations.

This is intentionally independent from the attack scorer.  It reads immutable
paper-result JSONL files, evaluates a small configuration matrix, and writes a
diagnostic report under the selected experiment run directory.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import struct
import sys
from typing import Any, Iterable, Mapping

import torch
import transformers
from transformers import SynthIDTextWatermarkLogitsProcessor

from .common import iter_jsonl, resolve_repo_path, run_root
from .detectors import _cached_snapshot, load_tokenizer


LEGACY_FIXED_KEYS = [
    673,
    197,
    281,
    206,
    634,
    513,
    697,
    187,
    876,
    555,
    837,
    271,
    897,
    455,
    314,
    494,
    236,
    539,
    394,
    414,
    531,
    108,
    285,
    596,
    820,
    219,
    312,
    183,
    392,
    972,
]

VARIANTS = (
    ("saved_seed_random_keys", "saved", "random"),
    ("saved_seed_fixed_keys", "saved", "fixed"),
    ("id_hash_seed_random_keys", "id_hash", "random"),
    ("id_hash_seed_fixed_keys", "id_hash", "fixed"),
)


def id_hash_seed(record_id: str) -> int:
    """Historical ``hash_str_to_int`` from ``src/_util.py``."""

    digest = hashlib.sha3_512(record_id.encode("utf-8")).digest()
    return int.from_bytes(digest[-8:], "big")


def random_keys(seed: int) -> list[int]:
    rng = random.Random(int(seed))
    return [int(rng.uniform(0, 1000)) for _ in range(30)]


def tensor_digest(values: torch.Tensor) -> str:
    flattened = values.detach().to("cpu").reshape(-1)
    if flattened.dtype == torch.bool or not flattened.dtype.is_floating_point:
        payload = bytes(int(value) & 0xFF for value in flattened.tolist())
    else:
        payload = b"".join(struct.pack("<d", float(value)) for value in flattened.tolist())
    return hashlib.sha256(payload).hexdigest()


def token_digest(input_ids: torch.Tensor) -> str:
    payload = b"".join(
        struct.pack("<q", int(token_id)) for token_id in input_ids.detach().cpu().reshape(-1)
    )
    return hashlib.sha256(payload).hexdigest()


def score_variant(
    input_ids: torch.Tensor,
    *,
    device: str,
    ngram_len: int,
    seed: int,
    keys: list[int],
) -> dict[str, Any]:
    config = {
        "ngram_len": int(ngram_len),
        "keys": list(keys),
        "sampling_table_size": 2**16,
        "sampling_table_seed": int(seed),
        "context_history_size": 1024,
    }
    processor = SynthIDTextWatermarkLogitsProcessor(device=device, **config)
    g_values = processor.compute_g_values(input_ids.to(device))
    observed = g_values.reshape(-1)
    count = int(observed.numel())
    successes = int(observed.to(torch.int64).sum().item())
    expected = count * 0.5
    standard_deviation = math.sqrt(count * 0.25) if count else 0.0
    z_score = (successes - expected) / standard_deviation if count else 0.0
    return {
        "seed": int(seed),
        "keys": list(keys),
        "keys_sha256": hashlib.sha256(
            b"".join(struct.pack("<q", int(key)) for key in keys)
        ).hexdigest(),
        "g_values_shape": list(g_values.shape),
        "g_values_sha256": tensor_digest(observed),
        "num_g_values": count,
        "num_ones": successes,
        "expected_ones": expected,
        "standard_deviation": standard_deviation,
        "z_score": z_score,
        "p_value": 0.5 * math.erfc(z_score / math.sqrt(2.0)),
        "formula": "(num_ones - 0.5 * num_g_values) / sqrt(0.25 * num_g_values)",
    }


def read_manifest(run_id: str) -> list[dict[str, Any]]:
    path = run_root(run_id) / "manifest.json"
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    return [dict(config) for config in manifest["configs"] if config["watermark"] == "synthid"]


def existing_scores(run_id: str, config_key: str) -> dict[str, dict[str, Any]]:
    path = run_root(run_id) / "scores" / f"{config_key}.jsonl.gz"
    if not path.is_file():
        return {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        rows = (json.loads(line) for line in handle if line.strip())
        return {str(row["task_name"]): row for row in rows}


def selected_records(config: Mapping[str, Any], limit: int) -> list[dict[str, Any]]:
    rows = iter_jsonl(resolve_repo_path(str(config["generate_path"])))
    result: list[dict[str, Any]] = []
    for row in rows:
        if row.get("z_score") is None:
            continue
        result.append(row)
        if len(result) == limit:
            break
    if len(result) != limit:
        raise RuntimeError(f"{config['key']} only supplied {len(result)} scored records")
    return result


def environment_report(device: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "requested_device": device,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
    }
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        result.update(
            {
                "cuda_device_index": index,
                "cuda_device_name": properties.name,
                "cuda_capability": list(torch.cuda.get_device_capability(index)),
            }
        )
    return result


def build_report(run_id: str, records_per_config: int, device: str) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    match_counts = {name: 0 for name, _, _ in VARIANTS}
    current_score_matches = 0

    for config in read_manifest(run_id):
        tokenizer = load_tokenizer(str(config["model_name"]))
        snapshot = _cached_snapshot(str(config["model_name"]))
        scored_rows = existing_scores(run_id, str(config["key"]))
        for record in selected_records(config, records_per_config):
            input_ids = tokenizer(str(record["g4d"]), return_tensors="pt").input_ids
            variants: dict[str, dict[str, Any]] = {}
            saved_z = float(record["z_score"])
            for name, seed_mode, key_mode in VARIANTS:
                seed = (
                    int(record["custom_seed"])
                    if seed_mode == "saved"
                    else id_hash_seed(str(record["id"]))
                )
                keys = random_keys(seed) if key_mode == "random" else list(LEGACY_FIXED_KEYS)
                result = score_variant(
                    input_ids,
                    device=device,
                    ngram_len=int(config["ngram_len"]),
                    seed=seed,
                    keys=keys,
                )
                result["saved_z_score"] = saved_z
                result["absolute_error_from_saved"] = abs(result["z_score"] - saved_z)
                result["exact_saved_match"] = math.isclose(
                    result["z_score"], saved_z, rel_tol=0.0, abs_tol=1e-12
                )
                match_counts[name] += int(result["exact_saved_match"])
                variants[name] = result

            prior = scored_rows.get(str(record["task_name"]), {})
            prior_baseline = prior.get("baseline_recomputed_score")
            prior_z = None if prior_baseline is None else float(prior_baseline["z_score"])
            current_z = float(variants["saved_seed_random_keys"]["z_score"])
            agrees_with_current_score = prior_z is not None and math.isclose(
                current_z, prior_z, rel_tol=0.0, abs_tol=1e-12
            )
            current_score_matches += int(agrees_with_current_score)
            cases.append(
                {
                    "config_key": config["key"],
                    "model_name": config["model_name"],
                    "tokenizer_snapshot": None if snapshot is None else str(snapshot),
                    "task_name": record["task_name"],
                    "record_id": record["id"],
                    "saved_custom_seed": int(record["custom_seed"]),
                    "historical_id_hash_seed": id_hash_seed(str(record["id"])),
                    "ngram_len": int(config["ngram_len"]),
                    "num_input_tokens": int(input_ids.numel()),
                    "input_ids_sha256": token_digest(input_ids),
                    "saved_z_score": saved_z,
                    "existing_current_baseline_z_score": prior_z,
                    "current_variant_matches_existing_score": agrees_with_current_score,
                    "variants": variants,
                }
            )

    total = len(cases)
    return {
        "schema_version": 1,
        "purpose": "Explain current SynthID arithmetic and identify the historical seed/key configuration.",
        "run_id": run_id,
        "records_per_config": records_per_config,
        "environment": environment_report(device),
        "current_implementation": {
            "variant": "saved_seed_random_keys",
            "tokenization": "tokenizer(g4d, return_tensors='pt').input_ids",
            "processor": "transformers.SynthIDTextWatermarkLogitsProcessor",
            "sampling_table_size": 2**16,
            "context_history_size": 1024,
            "coinflip_probability": 0.5,
        },
        "summary": {
            "case_count": total,
            "exact_saved_matches_by_variant": match_counts,
            "current_variant_matches_existing_score_count": current_score_matches,
            "current_variant_existing_score_comparison_count": sum(
                case["existing_current_baseline_z_score"] is not None for case in cases
            ),
        },
        "cases": cases,
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="rw100-useful-v1")
    parser.add_argument("--records-per-config", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.records_per_config < 1:
        parser.error("--records-per-config must be positive")
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    output = args.output or run_root(args.run_id) / "forensics" / "synthid_score_matrix.json"
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {output}; pass --overwrite")
    report = build_report(args.run_id, args.records_per_config, args.device)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(output)
    print(json.dumps(report["summary"], indent=2, sort_keys=True), flush=True)
    print(f"report={output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
