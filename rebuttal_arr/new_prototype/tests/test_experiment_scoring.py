from __future__ import annotations

import pytest

from experiment.detectors import get_synthid_config
from experiment.run import _check_saved_score, paper_auroc


EXPECTED_SYNTHID_KEYS = [
    673, 197, 281, 206, 634, 513, 697, 187, 876, 555,
    837, 271, 897, 455, 314, 494, 236, 539, 394, 414,
    531, 108, 285, 596, 820, 219, 312, 183, 392, 972,
]


def test_paper_auroc_uses_standard_normal_negative() -> None:
    assert paper_auroc([0.0] * 100) == pytest.approx(0.5, abs=0.01)
    assert paper_auroc([5.0] * 100) > 0.99


def test_synthid_uses_paper_fixed_keys_and_saved_sampling_seed() -> None:
    first = get_synthid_config(123, 5)
    second = get_synthid_config(456, 5)

    assert first["keys"] == EXPECTED_SYNTHID_KEYS
    assert second["keys"] == EXPECTED_SYNTHID_KEYS
    assert first["sampling_table_seed"] == 123
    assert second["sampling_table_seed"] == 456
    assert first["ngram_len"] == second["ngram_len"] == 5


def test_saved_score_regression_cannot_be_bypassed() -> None:
    class MismatchingScorer:
        def score(self, record, g4d):
            return {"z_score": 2.0}

    with pytest.raises(RuntimeError, match="saved-score regression failed"):
        _check_saved_score(
            MismatchingScorer(),
            {"key": "synthid-config"},
            {"task_name": "task/0", "g4d": "code", "z_score": 1.0},
        )
