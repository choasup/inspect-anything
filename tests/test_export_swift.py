import json

from inspect_one.data.export_swift import export_jsonl, to_swift
from inspect_one.data.format import Source, TrainingSample
from inspect_one.schema import Task
from inspect_one.teacher.base import extract_json


def ground_sample():
    return TrainingSample(
        sample_id="g1",
        image="img/shelf.jpg",
        task=Task.GROUND,
        target={
            "query": "可乐 330ml",
            "boxes": [
                {"x1": 0.1, "y1": 0.1, "x2": 0.3, "y2": 0.3, "label": "可乐 330ml", "score": 1.0}
            ],
        },
        source=Source.DISTILLED,
    )


def test_ground_export_structure():
    rec = to_swift(ground_sample())
    assert rec["images"] == ["img/shelf.jpg"]
    user, assistant = rec["messages"]
    assert user["role"] == "user" and user["content"].startswith("<image>")
    assert "可乐 330ml" in user["content"]
    answer = json.loads(assistant["content"])
    assert answer["boxes"][0]["x2"] == 0.3
    assert "query" not in answer  # query 在 prompt 里，不重复出现在监督目标里


def test_assistant_content_parses_with_inference_path():
    rec = to_swift(ground_sample())
    assert extract_json(rec["messages"][1]["content"])["boxes"][0]["label"] == "可乐 330ml"


def test_judge_export():
    s = TrainingSample(
        sample_id="j1",
        image="img/site.jpg",
        task=Task.JUDGE,
        target={"rule": "人员必须佩戴安全帽", "passed": False, "confidence": 0.9,
                "reason": "一人未佩戴", "evidence": []},
        source=Source.GOLD,
    )
    rec = to_swift(s)
    assert "人员必须佩戴安全帽" in rec["messages"][0]["content"]
    assert json.loads(rec["messages"][1]["content"])["passed"] is False


def test_compare_uses_two_image_tags():
    s = TrainingSample(
        sample_id="c1",
        image="img/actual.jpg",
        task=Task.COMPARE,
        target={"matches": False, "diffs": [], "reference_image": "img/ref.jpg",
                "instruction": "对比陈列与棚格图"},
        source=Source.DISTILLED,
    )
    rec = to_swift(s)
    assert rec["images"] == ["img/actual.jpg", "img/ref.jpg"]
    assert rec["messages"][0]["content"].count("<image>") == 2


def test_export_jsonl_round_trip(tmp_path):
    out = tmp_path / "train.jsonl"
    n = export_jsonl([ground_sample()], out)
    assert n == 1
    lines = out.read_text("utf-8").strip().splitlines()
    assert json.loads(lines[0])["images"] == ["img/shelf.jpg"]
