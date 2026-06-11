"""InspectBench 评测 runner。

数据集为 JSONL，每行一个样本：
  {"id": str, "subset": "retail|security|industrial", "task": "ground|count|judge",
   "image": str(路径，runner 不读取，透传给模型), "query"/"rule": str,
   "target": 任务相关的真值}

模型是一个 callable: (sample: dict) -> dict（结构与 schema 的对应 Result 一致），
本地学生、远程教师、任何 baseline 都能以同一接口接入评测。
评测集样本永不进训练集——去重靠 sample id 哈希，由数据管线另行强制。
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Callable

from inspect_one.bench.metrics import average_precision, count_mae, judge_f1
from inspect_one.schema import Box, ScoredBox

Model = Callable[[dict], dict]


def load_jsonl(path: str | Path) -> list[dict]:
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def evaluate(samples: list[dict], model: Model) -> dict:
    """返回 {subset: {metric: value, "n": 样本数}}，外加 "overall"。"""
    grouped: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    for sample in samples:
        grouped[sample["subset"]].append((sample, model(sample)))

    report: dict[str, dict] = {}
    for subset, pairs in grouped.items():
        metrics: dict[str, float] = {}
        ground = [(s, p) for s, p in pairs if s["task"] == "ground"]
        if ground:
            preds = [[ScoredBox(**b) for b in p.get("boxes", [])] for _, p in ground]
            gts = [[Box(**b) for b in s["target"]["boxes"]] for s, _ in ground]
            metrics["ap50"] = average_precision(preds, gts, iou_threshold=0.5)
        count = [(s, p) for s, p in pairs if s["task"] == "count"]
        if count:
            metrics["count_mae"] = count_mae(
                [int(p["count"]) for _, p in count],
                [int(s["target"]["count"]) for s, _ in count],
            )
        judge = [(s, p) for s, p in pairs if s["task"] == "judge"]
        if judge:
            metrics["judge_f1"] = judge_f1(
                [bool(p["passed"]) for _, p in judge],
                [bool(s["target"]["passed"]) for s, _ in judge],
            )
        metrics["n"] = len(pairs)
        report[subset] = metrics

    report["overall"] = {"n": len(samples)}
    return report


def regression_gate(
    current: dict, baseline: dict, max_drop: float = 0.01
) -> tuple[bool, list[str]]:
    """发布门禁（docs/03 第 5 节）：任一场景子集任一指标劣化 > max_drop 即阻断。

    误差类指标（*_mae）越小越好，其余越大越好。返回 (是否放行, 违规说明)。
    """
    violations = []
    for subset, base_metrics in baseline.items():
        for name, base_val in base_metrics.items():
            if name == "n" or subset not in current or name not in current[subset]:
                continue
            cur_val = current[subset][name]
            drop = cur_val - base_val if name.endswith("_mae") else base_val - cur_val
            if drop > max_drop:
                violations.append(
                    f"{subset}.{name}: {base_val:.4f} -> {cur_val:.4f}"
                )
    return (not violations, violations)
