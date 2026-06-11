"""任务原语的统一结构化 schema。

五类原语（ground / compare / count / vqa / judge）的输入输出在此定义，
教师层、学生层、飞轮标注、InspectBench 全部消费同一套类型。
坐标一律为相对图像宽高归一化的 xyxy，取值 [0, 1]。
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class Task(str, Enum):
    GROUND = "ground"
    COMPARE = "compare"
    COUNT = "count"
    VQA = "vqa"
    JUDGE = "judge"


class Box(BaseModel):
    x1: float = Field(ge=0.0, le=1.0)
    y1: float = Field(ge=0.0, le=1.0)
    x2: float = Field(ge=0.0, le=1.0)
    y2: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _ordered(self) -> "Box":
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError("box corners must satisfy x2>=x1, y2>=y1")
        return self

    @property
    def area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)


class ScoredBox(Box):
    label: str
    score: float = Field(ge=0.0, le=1.0)


class Point(BaseModel):
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)


class GroundResult(BaseModel):
    task: Task = Task.GROUND
    query: str
    boxes: list[ScoredBox] = Field(default_factory=list)


class CountResult(BaseModel):
    task: Task = Task.COUNT
    query: str
    count: int = Field(ge=0)
    points: list[Point] = Field(default_factory=list)


class CompareDiff(BaseModel):
    region: Box
    expected: str
    actual: str


class CompareResult(BaseModel):
    task: Task = Task.COMPARE
    matches: bool
    diffs: list[CompareDiff] = Field(default_factory=list)


class VQAResult(BaseModel):
    task: Task = Task.VQA
    question: str
    answer: str
    region: Optional[Box] = None


class JudgeResult(BaseModel):
    task: Task = Task.JUDGE
    rule: str
    passed: bool
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    evidence: list[ScoredBox] = Field(default_factory=list)
    reason: str = ""


PRIMITIVE_RESULTS = {
    Task.GROUND: GroundResult,
    Task.COUNT: CountResult,
    Task.COMPARE: CompareResult,
    Task.VQA: VQAResult,
    Task.JUDGE: JudgeResult,
}
