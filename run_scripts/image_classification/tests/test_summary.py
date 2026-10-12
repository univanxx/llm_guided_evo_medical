"""Check the reported metrics with invalid answers in both true classes."""
import pytest
from summarize_results import score


def test_invalid_answers_remain_in_class_denominators():
    """Use hand-computed confusion counts, including an invalid pneumonia prediction."""
    rows = [{"true_id": 0, "pred_id": 0}, {"true_id": 0, "pred_id": 1}, {"true_id": 1, "pred_id": 1}, {"true_id": 1}]
    metrics = score(rows)
    assert metrics["accuracy"] == 0.5
    assert metrics["balanced_accuracy"] == 0.5
    assert metrics["f1_pneumonia"] == 0.5
    assert metrics["macro_f1"] == pytest.approx((2 / 3 + 0.5) / 2)
    assert metrics["invalid_rate"] == 0.25


def test_unbalanced_test_accuracy_differs_from_balanced_accuracy():
    """Keep test accuracy distinct from mean recall when class frequencies differ."""
    rows = [{"true_id": 0, "pred_id": 0}, {"true_id": 1, "pred_id": 0}, {"true_id": 1, "pred_id": 0}]
    assert score(rows)["accuracy"] == pytest.approx(1 / 3)
    assert score(rows)["balanced_accuracy"] == 0.5
    assert score(rows)["f1_pneumonia"] == 0
