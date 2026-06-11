import json
from pathlib import Path

from inspect_one.data.validate import validate_file

EXAMPLES = Path(__file__).parent.parent / "data" / "examples"


def test_distilled_example_file_is_valid():
    assert validate_file(EXAMPLES / "distilled.example.jsonl") == []


def test_bench_example_file_is_valid():
    assert validate_file(EXAMPLES / "bench.example.jsonl", bench=True) == []


def test_reports_line_numbers_for_bad_rows(tmp_path):
    good = json.loads((EXAMPLES / "distilled.example.jsonl").read_text().splitlines()[0])
    bad = {**good, "sample_id": "x"}
    bad["target"] = {"query": "q", "boxes": [{"x1": 0.5, "y1": 0.1, "x2": 0.2, "y2": 0.3, "label": "l", "score": 1.0}]}  # 角点顺序非法
    p = tmp_path / "d.jsonl"
    p.write_text(
        json.dumps(good, ensure_ascii=False) + "\n" + json.dumps(bad, ensure_ascii=False) + "\n",
        "utf-8",
    )
    errors = validate_file(p)
    assert len(errors) == 1 and errors[0].startswith("line 2:")


def test_detects_duplicate_ids(tmp_path):
    line = (EXAMPLES / "distilled.example.jsonl").read_text().splitlines()[0]
    p = tmp_path / "d.jsonl"
    p.write_text(line + "\n" + line + "\n", "utf-8")
    errors = validate_file(p)
    assert len(errors) == 1 and "duplicate" in errors[0]


def test_bench_requires_task_input_field(tmp_path):
    row = {"id": "b1", "subset": "retail", "task": "ground", "image": "a.jpg",
           "target": {"boxes": []}}  # 缺 query
    p = tmp_path / "b.jsonl"
    p.write_text(json.dumps(row) + "\n", "utf-8")
    errors = validate_file(p, bench=True)
    assert len(errors) == 1 and "query" in errors[0]
