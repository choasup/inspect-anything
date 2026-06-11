import pytest

from inspect_one.bench.metrics import (
    average_precision,
    count_mae,
    fpr_at_recall,
    iou,
    judge_f1,
)
from inspect_one.schema import Box, ScoredBox


def box(x1, y1, x2, y2):
    return Box(x1=x1, y1=y1, x2=x2, y2=y2)


def sbox(x1, y1, x2, y2, score=0.9, label="obj"):
    return ScoredBox(x1=x1, y1=y1, x2=x2, y2=y2, score=score, label=label)


def test_iou_identical_and_disjoint():
    a = box(0.1, 0.1, 0.5, 0.5)
    assert iou(a, a) == pytest.approx(1.0)
    assert iou(a, box(0.6, 0.6, 0.9, 0.9)) == 0.0


def test_iou_half_overlap():
    a = box(0.0, 0.0, 0.4, 0.4)
    b = box(0.2, 0.0, 0.6, 0.4)
    assert iou(a, b) == pytest.approx(1 / 3)


def test_ap_perfect_detection():
    gts = [[box(0.1, 0.1, 0.3, 0.3)], [box(0.5, 0.5, 0.8, 0.8)]]
    preds = [[sbox(0.1, 0.1, 0.3, 0.3)], [sbox(0.5, 0.5, 0.8, 0.8)]]
    assert average_precision(preds, gts) == pytest.approx(1.0)


def test_ap_false_positive_lowers_score():
    gts = [[box(0.1, 0.1, 0.3, 0.3)]]
    preds = [[sbox(0.1, 0.1, 0.3, 0.3, score=0.8), sbox(0.6, 0.6, 0.9, 0.9, score=0.9)]]
    ap = average_precision(preds, gts)
    assert 0.0 < ap < 1.0


def test_ap_duplicate_predictions_not_double_counted():
    gts = [[box(0.1, 0.1, 0.3, 0.3)]]
    preds = [[sbox(0.1, 0.1, 0.3, 0.3, score=0.9), sbox(0.1, 0.1, 0.3, 0.3, score=0.8)]]
    assert average_precision(preds, gts) == pytest.approx(1.0)


def test_count_mae():
    assert count_mae([10, 20], [12, 20]) == pytest.approx(1.0)
    assert count_mae([], []) == 0.0


def test_judge_f1_violation_as_positive():
    # 模型全说通过、实际有违规 → F1 = 0
    assert judge_f1([True, True], [False, True]) == 0.0
    assert judge_f1([False, True], [False, True]) == pytest.approx(1.0)


def test_fpr_at_recall_perfect_separation():
    scores = [0.9, 0.8, 0.2, 0.1]
    labels = [True, True, False, False]
    assert fpr_at_recall(scores, labels, target_recall=1.0) == 0.0


def test_fpr_at_recall_pays_for_high_recall():
    # 要捞回 score=0.1 的正样本就必须把两个更高分的负样本一起放进来
    scores = [0.9, 0.5, 0.4, 0.1]
    labels = [True, False, False, True]
    assert fpr_at_recall(scores, labels, target_recall=1.0) == pytest.approx(1.0)
    assert fpr_at_recall(scores, labels, target_recall=0.5) == 0.0
