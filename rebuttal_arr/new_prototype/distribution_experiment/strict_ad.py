#!/usr/bin/env python3
"""Run the fixed-N(0,1) AD diagnostic on an existing scored run."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json

import numpy as np

from experiment.common import atomic_json, iter_jsonl

from .run import fixed_ad_null, fixed_standard_normal_ad, load_manifest, run_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--draws", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=10_771)
    args = parser.parse_args()

    root = run_root(args.run_id)
    manifest = load_manifest(args.run_id)
    expected_samples = int(manifest["walk"]["trajectories_per_seed"])
    candidates = {
        str(candidate["candidate_key"]): candidate
        for candidate in manifest["candidates"]
    }
    scores: dict[str, list[float]] = defaultdict(list)
    for path in sorted((root / "scores").glob("*.jsonl.gz")):
        for row in iter_jsonl(path):
            scores[str(row["candidate_key"])].append(float(row["score"]["z_score"]))
    if set(scores) != set(candidates):
        raise RuntimeError("scored candidate set does not match manifest")

    null = fixed_ad_null(args.draws, expected_samples, args.seed)
    critical_5 = float(np.quantile(null, 0.95))
    critical_15 = float(np.quantile(null, 0.85))
    rows = []
    for key in sorted(scores):
        values = scores[key]
        if len(values) != expected_samples:
            raise RuntimeError(f"{key}: {len(values)} scores != {expected_samples}")
        statistic = fixed_standard_normal_ad(values)
        p_value = float((1 + np.count_nonzero(null >= statistic)) / (args.draws + 1))
        candidate = candidates[key]
        rows.append(
            {
                "candidate_key": key,
                "watermark": candidate["watermark"],
                "task_name": candidate["task_name"],
                "samples": len(values),
                "sample_mean": float(np.mean(values)),
                "sample_std": float(np.std(values, ddof=1)),
                "strict_ad_statistic": statistic,
                "strict_ad_p_value": p_value,
                "strict_ad_accept_5": bool(statistic <= critical_5),
                "strict_ad_accept_15": bool(statistic <= critical_15),
            }
        )
    result = {
        "run_id": args.run_id,
        "definition": "fixed N(0,1) Anderson-Darling with Monte Carlo null",
        "monte_carlo_draws": args.draws,
        "seed": args.seed,
        "critical_5": critical_5,
        "critical_15": critical_15,
        "spaces": len(rows),
        "accept_5": sum(bool(row["strict_ad_accept_5"]) for row in rows),
        "accept_rate_5": float(np.mean([row["strict_ad_accept_5"] for row in rows])),
        "accept_15": sum(bool(row["strict_ad_accept_15"]) for row in rows),
        "accept_rate_15": float(np.mean([row["strict_ad_accept_15"] for row in rows])),
        "rows": rows,
    }
    atomic_json(root / "strict_ad.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
