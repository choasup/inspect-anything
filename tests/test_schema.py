import pytest
from pydantic import ValidationError

from inspect_one.schema import Box, CountResult, JudgeResult, ScoredBox


def test_box_rejects_inverted_corners():
    with pytest.raises(ValidationError):
        Box(x1=0.5, y1=0.1, x2=0.2, y2=0.3)


def test_box_rejects_out_of_range():
    with pytest.raises(ValidationError):
        Box(x1=0.0, y1=0.0, x2=1.2, y2=0.5)


def test_box_area():
    assert Box(x1=0.0, y1=0.0, x2=0.5, y2=0.5).area == pytest.approx(0.25)


def test_count_rejects_negative():
    with pytest.raises(ValidationError):
        CountResult(query="bottles", count=-1)


def test_judge_round_trip():
    j = JudgeResult(
        rule="所有员工必须佩戴安全帽",
        passed=False,
        confidence=0.93,
        evidence=[ScoredBox(x1=0.1, y1=0.1, x2=0.3, y2=0.4, label="未戴安全帽", score=0.9)],
        reason="左侧一名员工未佩戴安全帽",
    )
    restored = JudgeResult.model_validate_json(j.model_dump_json())
    assert restored == j
