import json

import pytest

from inspect_one.bench.runner import evaluate, load_jsonl, regression_gate

SAMPLES = [
    {
        "id": "r1",
        "subset": "retail",
        "task": "ground",
        "image": "img/r1.jpg",
        "query": "可乐 330ml",
        "target": {"boxes": [{"x1": 0.1, "y1": 0.1, "x2": 0.3, "y2": 0.3}]},
    },
    {
        "id": "r2",
        "subset": "retail",
        "task": "count",
        "image": "img/r2.jpg",
        "query": "货架上的瓶装水",
        "target": {"count": 12},
    },
    {
        "id": "s1",
        "subset": "security",
        "task": "judge",
        "image": "img/s1.jpg",
        "rule": "工地人员必须佩戴安全帽",
        "target": {"passed": False},
    },
]


def perfect_model(sample):
    task = sample["task"]
    if task == "ground":
        return {
            "boxes": [
                {**b, "label": sample["query"], "score": 0.95}
                for b in sample["target"]["boxes"]
            ]
        }
    if task == "count":
        return {"count": sample["target"]["count"]}
    return {"passed": sample["target"]["passed"]}


def test_evaluate_perfect_model():
    report = evaluate(SAMPLES, perfect_model)
    assert report["retail"]["ap50"] == pytest.approx(1.0)
    assert report["retail"]["count_mae"] == 0.0
    assert report["security"]["judge_f1"] == pytest.approx(1.0)
    assert report["overall"]["n"] == 3


def test_load_jsonl_round_trip(tmp_path):
    p = tmp_path / "bench.jsonl"
    p.write_text("\n".join(json.dumps(s, ensure_ascii=False) for s in SAMPLES), "utf-8")
    assert load_jsonl(p) == SAMPLES


def test_regression_gate_blocks_subset_drop():
    baseline = {"retail": {"ap50": 0.80, "n": 100}, "security": {"judge_f1": 0.90, "n": 50}}
    worse = {"retail": {"ap50": 0.75, "n": 100}, "security": {"judge_f1": 0.91, "n": 50}}
    ok, violations = regression_gate(worse, baseline, max_drop=0.01)
    assert not ok
    assert violations == ["retail.ap50: 0.8000 -> 0.7500"]


def test_regression_gate_mae_direction():
    baseline = {"retail": {"count_mae": 1.0, "n": 100}}
    better = {"retail": {"count_mae": 0.5, "n": 100}}
    worse = {"retail": {"count_mae": 1.5, "n": 100}}
    assert regression_gate(better, baseline)[0]
    assert not regression_gate(worse, baseline)[0]
