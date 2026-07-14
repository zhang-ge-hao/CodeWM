#!/usr/bin/env python3
"""Compare fixed-N(0,1) accepted and rejected pilot spaces."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import gzip
import json
from pathlib import Path
import statistics
import sys
from typing import Any

import numpy as np
from scipy import stats

from distribution_experiment.run import iter_transform_rows, load_manifest, record_for_candidate, run_root
from experiment.common import iter_jsonl


ADVANCED = {
    "insert_true_opaque_guard",
    "remove_true_opaque_guard",
    "insert_false_opaque_guard",
    "remove_false_opaque_guard",
    "flatten_straight_line",
    "restore_straight_line",
    "flatten_simple_if",
    "restore_simple_if",
}


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def load_features(run_id: str) -> list[dict[str, Any]]:
    root = run_root(run_id)
    manifest = load_manifest(run_id)
    spaces = {
        row["candidate_key"]: row
        for row in csv.DictReader((root / "spaces.csv").open(encoding="utf-8"))
    }
    transforms: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in iter_transform_rows(run_id):
        transforms[str(row["candidate_key"])].append(row)
    scores: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted((root / "scores").glob("*.jsonl.gz")):
        for row in iter_jsonl(path):
            scores[str(row["candidate_key"])].append(row)

    result: list[dict[str, Any]] = []
    for candidate in manifest["candidates"]:
        key = str(candidate["candidate_key"])
        source = record_for_candidate(candidate)
        ts = transforms[key]
        ss = scores[key]
        rule_counts: Counter[str] = Counter()
        for row in ts:
            rule_counts.update({str(k): int(v) for k, v in row.get("rule_counts", {}).items()})
        token_counts = [
            float(row["score"]["num_tokens_scored"])
            for row in ss
            if row.get("score", {}).get("num_tokens_scored") is not None
        ]
        source_bytes = len(str(source["solution"]).encode("utf-8"))
        row = spaces[key]
        result.append(
            {
                "candidate_key": key,
                "accepted": row["fixed_ad_accept_5"] == "True",
                "watermark": candidate["watermark"],
                "model_slug": candidate["model_slug"],
                "dataset": candidate["dataset"],
                "original_z": float(candidate["original_z_score"]),
                "source_bytes": float(source_bytes),
                "source_detection_chars": float(len(str(source["g4d"]))),
                "source_s_len": float(source["s_len"]),
                "endpoint_bytes": mean([float(r["final"]["bytes"]) for r in ts]),
                "endpoint_growth": mean([float(r["final"]["bytes"]) / source_bytes for r in ts]),
                "endpoint_detection_chars": mean([float(len(r["detection_g4d"])) for r in ts]),
                "endpoint_tokens_scored": mean(token_counts),
                "identity_steps": float(rule_counts["identity"]) / len(ts),
                "rename_steps": float(rule_counts["rename_variable"]) / len(ts),
                "comment_steps": float(
                    sum(value for rule, value in rule_counts.items() if rule.startswith("replace_comment"))
                ) / len(ts),
                "advanced_steps": float(sum(rule_counts[rule] for rule in ADVANCED)) / len(ts),
                "sample_mean": float(row["sample_mean"]),
                "sample_std": float(row["sample_std"]),
                "fixed_ad_statistic": float(row["fixed_ad_statistic"]),
                "fixed_ad_p_value": float(row["fixed_ad_p_value"]),
            }
        )
    return result


NUMERIC = (
    "original_z",
    "source_bytes",
    "source_detection_chars",
    "source_s_len",
    "endpoint_bytes",
    "endpoint_growth",
    "endpoint_detection_chars",
    "endpoint_tokens_scored",
    "identity_steps",
    "rename_steps",
    "comment_steps",
    "advanced_steps",
)


def compare(rows: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = [row for row in rows if row["accepted"]]
    rejected = [row for row in rows if not row["accepted"]]
    result: dict[str, Any] = {
        "spaces": len(rows),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "numeric": {},
    }
    for feature in NUMERIC:
        a = np.asarray([row[feature] for row in accepted], dtype=float)
        r = np.asarray([row[feature] for row in rejected], dtype=float)
        if len(a) and len(r):
            mw = stats.mannwhitneyu(a, r, alternative="two-sided")
            rank_biserial = 2.0 * float(mw.statistic) / (len(a) * len(r)) - 1.0
        else:
            mw = None
            rank_biserial = float("nan")
        all_values = np.asarray([row[feature] for row in rows], dtype=float)
        ad_values = np.asarray([row["fixed_ad_statistic"] for row in rows], dtype=float)
        corr = stats.spearmanr(all_values, ad_values) if len(set(all_values)) > 1 else None
        result["numeric"][feature] = {
            "accepted_mean": mean(a.tolist()),
            "accepted_median": float(np.median(a)) if len(a) else None,
            "rejected_mean": mean(r.tolist()),
            "rejected_median": float(np.median(r)) if len(r) else None,
            "mann_whitney_p": None if mw is None else float(mw.pvalue),
            "rank_biserial_accepted_minus_rejected": rank_biserial,
            "spearman_with_fixed_ad_statistic": None if corr is None else float(corr.statistic),
            "spearman_p": None if corr is None else float(corr.pvalue),
        }
    return result


def categorical(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    values: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        values[str(row[field])].append(row)
    return {
        key: {
            "spaces": len(group),
            "accepted": sum(bool(row["accepted"]) for row in group),
            "accept_rate": mean([float(row["accepted"]) for row in group]),
        }
        for key, group in sorted(values.items())
    }


def length_quartiles(rows: list[dict[str, Any]], feature: str) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: (float(row[feature]), str(row["candidate_key"])))
    groups = np.array_split(np.asarray(ordered, dtype=object), 4)
    return [
        {
            "quartile": index + 1,
            "spaces": len(group),
            "min": float(min(row[feature] for row in group)),
            "max": float(max(row[feature] for row in group)),
            "median": float(np.median([row[feature] for row in group])),
            "accepted": sum(bool(row["accepted"]) for row in group),
            "accept_rate": mean([float(row["accepted"]) for row in group]),
        }
        for index, group in enumerate(groups)
    ]


def main() -> None:
    run_id = sys.argv[1] if len(sys.argv) > 1 else "rw100-z4-sample100-v2"
    rows = load_features(run_id)
    output = {
        "run_id": run_id,
        "overall": compare(rows),
        "by_scheme": {
            scheme: compare([row for row in rows if row["watermark"] == scheme])
            for scheme in ("wllm", "sweet", "synthid")
        },
        "categorical": {
            field: categorical(rows, field)
            for field in ("watermark", "model_slug", "dataset")
        },
        "source_length_quartiles": length_quartiles(rows, "source_bytes"),
        "endpoint_length_quartiles": length_quartiles(rows, "endpoint_bytes"),
    }
    path = run_root(run_id) / "feature_analysis.json"
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
