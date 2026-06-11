"""InspectBench 指标实现（docs/04 第 1 节）。

- 检测：IoU 贪心匹配 → AP（precision 包络下面积）
- 计数：MAE
- 审计/judge：F1
- 安防核心指标：固定召回下的误报率 FPR@recall
"""

from __future__ import annotations

from inspect_one.schema import Box, ScoredBox


def iou(a: Box, b: Box) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def average_precision(
    predictions: list[list[ScoredBox]],
    targets: list[list[Box]],
    iou_threshold: float = 0.5,
) -> float:
    """单类别 AP。predictions/targets 按图像对齐；预测全图汇总按分数降序贪心匹配。"""
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must align per image")
    n_gt = sum(len(t) for t in targets)
    if n_gt == 0:
        return 0.0

    flat: list[tuple[float, int, ScoredBox]] = [
        (p.score, img_idx, p)
        for img_idx, preds in enumerate(predictions)
        for p in preds
    ]
    flat.sort(key=lambda t: t[0], reverse=True)

    matched: list[set[int]] = [set() for _ in targets]
    tps: list[int] = []
    for _, img_idx, pred in flat:
        gts = targets[img_idx]
        best_iou, best_j = 0.0, -1
        for j, gt in enumerate(gts):
            if j in matched[img_idx]:
                continue
            v = iou(pred, gt)
            if v > best_iou:
                best_iou, best_j = v, j
        if best_iou >= iou_threshold:
            matched[img_idx].add(best_j)
            tps.append(1)
        else:
            tps.append(0)

    # precision 包络下的面积（COCO 连续 AP 的简化形式）
    ap, tp_cum = 0.0, 0
    precisions, recalls = [], []
    for i, tp in enumerate(tps, start=1):
        tp_cum += tp
        precisions.append(tp_cum / i)
        recalls.append(tp_cum / n_gt)
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    prev_recall = 0.0
    for p, r in zip(precisions, recalls):
        ap += p * (r - prev_recall)
        prev_recall = r
    return ap


def count_mae(predicted: list[int], actual: list[int]) -> float:
    if len(predicted) != len(actual):
        raise ValueError("predicted and actual must align")
    if not predicted:
        return 0.0
    return sum(abs(p - a) for p, a in zip(predicted, actual)) / len(predicted)


def judge_f1(predicted: list[bool], actual: list[bool]) -> float:
    """judge 任务的 F1，违规（False=不通过）视为正类。"""
    tp = sum(1 for p, a in zip(predicted, actual) if not p and not a)
    fp = sum(1 for p, a in zip(predicted, actual) if not p and a)
    fn = sum(1 for p, a in zip(predicted, actual) if p and not a)
    denom = 2 * tp + fp + fn
    return 2 * tp / denom if denom else 0.0


def fpr_at_recall(
    scores: list[float], labels: list[bool], target_recall: float = 0.99
) -> float:
    """固定召回下的误报率：取满足 recall >= target 的最高阈值，返回该点 FPR。

    labels: True = 真实告警事件。无法达到目标召回时返回 1.0（最差）。
    """
    if len(scores) != len(labels):
        raise ValueError("scores and labels must align")
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0:
        raise ValueError("need at least one positive")

    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    tp = fp = 0
    best_fpr = None
    for idx in order:
        if labels[idx]:
            tp += 1
        else:
            fp += 1
        if tp / n_pos >= target_recall:
            fpr = fp / n_neg if n_neg else 0.0
            best_fpr = fpr if best_fpr is None else min(best_fpr, fpr)
    return 1.0 if best_fpr is None else best_fpr
