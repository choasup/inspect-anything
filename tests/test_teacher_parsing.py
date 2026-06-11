import pytest

from inspect_one.teacher.base import extract_json


def test_extracts_fenced_json_block():
    text = '前面有解释文字。\n```json\n{"count": 3, "points": []}\n```\n后面还有话。'
    assert extract_json(text) == {"count": 3, "points": []}


def test_extracts_bare_object_with_nesting():
    text = 'Sure! {"boxes": [{"x1": 0.1, "y1": 0.2, "x2": 0.3, "y2": 0.4}]} done'
    assert extract_json(text)["boxes"][0]["x2"] == 0.3


def test_extracts_bare_array():
    assert extract_json("[1, 2, 3]") == [1, 2, 3]


def test_raises_on_no_json():
    with pytest.raises(ValueError):
        extract_json("这张图里没有检测到目标。")
