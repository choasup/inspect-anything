"""教师层抽象：任何能完成五类原语的后端（自托管 VLM、API、SAM3 服务）都实现 Teacher。

教师输出必须是 schema 中的结构化类型；自由文本到结构化的解析也集中在这里，
学生蒸馏与飞轮标注共用同一解析路径，保证训练目标格式一致。
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any

from inspect_one.schema import (
    CompareResult,
    CountResult,
    GroundResult,
    JudgeResult,
    VQAResult,
)


class Teacher(ABC):
    @abstractmethod
    def ground(self, image: bytes, query: str) -> GroundResult: ...

    @abstractmethod
    def count(self, image: bytes, query: str) -> CountResult: ...

    @abstractmethod
    def compare(self, image: bytes, reference: bytes, instruction: str) -> CompareResult: ...

    @abstractmethod
    def vqa(self, image: bytes, question: str) -> VQAResult: ...

    @abstractmethod
    def judge(self, image: bytes, rule: str) -> JudgeResult: ...


_JSON_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> Any:
    """从模型回复中提取 JSON：优先 ```json``` 代码块，否则取首个平衡的 {} / [] 片段。"""
    m = _JSON_BLOCK.search(text)
    if m:
        return json.loads(m.group(1))
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = text.find(open_ch)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(text)):
            if text[i] == open_ch:
                depth += 1
            elif text[i] == close_ch:
                depth -= 1
                if depth == 0:
                    return json.loads(text[start : i + 1])
    raise ValueError(f"no JSON found in model output: {text[:200]!r}")
