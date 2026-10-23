"""AUROC aggregation for the empirical-negative rebuttal experiment."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

from sklearn.metrics import roc_auc_score


POSITIVE_ORDER = (
    "clean_wllm",
    "pyminify_wllm",
    "pyminifier_wllm",
)

NEGATIVE_ORDER = (
    "clean_no_wm_llm",
    "benchmark_reference",
    "pyminify_no_wm_llm",
    "pyminifier_no_wm_llm",
)

OBF_NAME_TO_POSITIVE = {
    "Original": "clean_wllm",
    "pyminify": "pyminify_wllm",
    "pyminifier": "pyminifier_wllm",
}


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _task_set_hash(task_ids: Iterable[str]) -> str:
    payload = "\n".join(sorted(task_ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def calculate_empirical_auroc(positive_scores: list[float], negative_scores: list[float]) -> float:
    """Calculate AUROC with larger z-scores denoting the positive class."""

    if not positive_scores:
        raise ValueError("Cannot calculate AUROC without positive scores.")
    if not negative_scores:
        raise ValueError("Cannot calculate AUROC without negative scores.")
    labels = [1] * len(positive_scores) + [0] * len(negative_scores)
    scores = positive_scores + negative_scores
    return float(roc_auc_score(labels, scores))


def aggregate_record_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create the complete four-negative by three-positive AUROC matrix.

    The paper's positive cohort is reconstructed by retaining tasks for which
    all three saved positive variants have a finite z-score.  This positive
    cohort is fixed across all negative classes.  Missing negative scores only
    reduce the corresponding negative distribution.
    """

    if not rows:
        raise ValueError("No score records were provided.")

    first = rows[0]
    detector = first["detector"]
    dataset = first["dataset"]
    config_id = first["config"]

    for row in rows:
        if row["dataset"] != dataset or row["config"] != config_id:
            raise ValueError("A record file must contain one dataset/config pair.")

    retained_rows = []
    dropped_positive: dict[str, int] = {name: 0 for name in POSITIVE_ORDER}
    for row in rows:
        missing = [
            name
            for name in POSITIVE_ORDER
            if not _finite_number(row.get("positive", {}).get(name, {}).get("z_score"))
        ]
        if missing:
            for name in missing:
                dropped_positive[name] += 1
            continue
        retained_rows.append(row)

    if not retained_rows:
        raise ValueError(f"No complete positive cohort for {dataset}/{config_id}.")

    positive_task_ids = [row["task"] for row in retained_rows]
    positive_task_hash = _task_set_hash(positive_task_ids)
    result: list[dict[str, Any]] = []

    # Negative-major ordering is intentional: the experiment changes the
    # negative class while evaluating every already-existing positive variant.
    for negative_name in NEGATIVE_ORDER:
        negative_rows = [
            row
            for row in retained_rows
            if _finite_number(row.get("negative", {}).get(negative_name, {}).get("z_score"))
        ]
        negative_scores = [
            float(row["negative"][negative_name]["z_score"])
            for row in negative_rows
        ]
        negative_task_ids = [row["task"] for row in negative_rows]

        for positive_name in POSITIVE_ORDER:
            positive_scores = [
                float(row["positive"][positive_name]["z_score"])
                for row in retained_rows
            ]
            result.append(
                {
                    "dataset": dataset,
                    "config": config_id,
                    "delta": detector["delta"],
                    "gamma": detector["gamma"],
                    "temperature": detector["temperature"],
                    "ngram_len": detector["ngram_len"],
                    "negative": negative_name,
                    "positive": positive_name,
                    "auroc": calculate_empirical_auroc(positive_scores, negative_scores),
                    "n_positive": len(positive_scores),
                    "n_negative": len(negative_scores),
                    "positive_task_set_sha256": positive_task_hash,
                    "negative_task_set_sha256": _task_set_hash(negative_task_ids),
                    "dropped_positive": {
                        key: value for key, value in dropped_positive.items() if value
                    },
                    "dropped_negative": {
                        "missing_or_failed": len(retained_rows) - len(negative_rows)
                    }
                    if len(negative_rows) != len(retained_rows)
                    else {},
                    "source": "new",
                }
            )

    return result

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def existing_normal_rows(
    dataset: str,
    config_id: str,
    metrics_path: Path,
) -> list[dict[str, Any]]:
    """Normalize the paper's already-computed N(0,1) AUROCs."""

    rows = load_jsonl(metrics_path)
    result = []
    for row in rows:
        positive_name = OBF_NAME_TO_POSITIVE.get(row.get("obf_name"))
        if positive_name is None:
            continue
        result.append(
            {
                "dataset": dataset,
                "config": config_id,
                "delta": row.get("delta"),
                "gamma": row.get("gamma"),
                "temperature": row.get("temperature"),
                "ngram_len": row.get("ngram_len"),
                "negative": "standard_normal",
                "positive": positive_name,
                "auroc": row.get("auroc"),
                "n_positive": row.get("comp_c"),
                "n_negative": None,
                "source": "existing",
            }
        )
    if len(result) != 3:
        raise ValueError(
            f"Expected three existing positive variants in {metrics_path}, got {len(result)}."
        )
    return result
