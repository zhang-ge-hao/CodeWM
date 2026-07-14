from __future__ import annotations

from pathlib import Path
import sys
import unittest


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from aggregate import (  # noqa: E402
    NEGATIVE_ORDER,
    POSITIVE_ORDER,
    aggregate_record_rows,
    calculate_empirical_auroc,
)


def _row(task: str, offset: float, missing_negative: str | None = None) -> dict:
    negative = {
        name: {"z_score": -1.0 + offset}
        for name in NEGATIVE_ORDER
    }
    if missing_negative:
        negative[missing_negative]["z_score"] = None
    return {
        "dataset": "humaneval_py",
        "config": "001",
        "task": task,
        "detector": {
            "delta": 0.5,
            "gamma": 0.1,
            "temperature": 1.0,
            "ngram_len": 5,
        },
        "positive": {
            name: {"z_score": 1.0 + offset}
            for name in POSITIVE_ORDER
        },
        "negative": negative,
    }


class AggregateTest(unittest.TestCase):
    def test_auc_ties_and_direction(self) -> None:
        self.assertEqual(calculate_empirical_auroc([2.0, 3.0], [0.0, 1.0]), 1.0)
        self.assertEqual(calculate_empirical_auroc([0.0, 1.0], [2.0, 3.0]), 0.0)
        self.assertEqual(calculate_empirical_auroc([1.0], [1.0]), 0.5)

    def test_complete_negative_by_positive_matrix(self) -> None:
        rows = [
            _row("humaneval_py/0", 0.0),
            _row("humaneval_py/1", 0.2, "pyminify_no_wm_llm"),
        ]
        metrics = aggregate_record_rows(rows)
        self.assertEqual(len(metrics), 12)
        self.assertEqual(
            [(row["negative"], row["positive"]) for row in metrics],
            [
                (negative, positive)
                for negative in NEGATIVE_ORDER
                for positive in POSITIVE_ORDER
            ],
        )
        clean = [row for row in metrics if row["negative"] == "clean_no_wm_llm"]
        self.assertTrue(all(row["n_positive"] == 2 for row in clean))
        self.assertTrue(all(row["n_negative"] == 2 for row in clean))
        pyminify = [
            row for row in metrics if row["negative"] == "pyminify_no_wm_llm"
        ]
        self.assertTrue(all(row["n_positive"] == 2 for row in pyminify))
        self.assertTrue(all(row["n_negative"] == 1 for row in pyminify))


if __name__ == "__main__":
    unittest.main()
