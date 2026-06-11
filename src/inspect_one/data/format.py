"""统一训练样本格式。

训练数据有三个来源，全部转换成同一种 TrainingSample 落盘（JSONL，版本化）：
- PUBLIC：公开数据集（Objects365/GoldG/COCO/RefCOCO 等），经 converters 转换
- DISTILLED：教师离线蒸馏产物（一次生成、复用，训练时不在线调用教师）
- GOLD：飞轮人审金标

监督目标复用 schema.py 的原语 Result 类型，保证训练目标格式 == 推理输出格式。
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, field_validator, model_validator

from inspect_one.schema import PRIMITIVE_RESULTS, Task


class Source(str, Enum):
    PUBLIC = "public"
    DISTILLED = "distilled"
    GOLD = "gold"


class TrainingSample(BaseModel):
    sample_id: str
    image: str  # 图像路径/URI，数据加载器负责读取
    task: Task
    target: dict  # 对应原语的 Result 负载，按 task 校验
    source: Source

    @field_validator("sample_id", "image")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must be non-empty")
        return v

    @model_validator(mode="after")
    def _target_matches_task(self) -> "TrainingSample":
        PRIMITIVE_RESULTS[self.task].model_validate({**self.target, "task": self.task})
        return self
