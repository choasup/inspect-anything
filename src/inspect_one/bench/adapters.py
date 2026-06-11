"""把任何 Teacher 实现适配成 InspectBench 的 Model 接口。

swift deploy 起的学生模型服务是 OpenAI 兼容接口，用 QwenVLTeacher 指向它
即可作为被评测模型——教师零样本基线和学生走同一条评测路径。
"""

from __future__ import annotations

from pathlib import Path

from inspect_one.bench.runner import Model
from inspect_one.teacher.base import Teacher


def teacher_as_model(teacher: Teacher, image_root: str | Path = ".") -> Model:
    root = Path(image_root)

    def model(sample: dict) -> dict:
        image = (root / sample["image"]).read_bytes()
        task = sample["task"]
        if task == "ground":
            return teacher.ground(image, sample["query"]).model_dump()
        if task == "count":
            return teacher.count(image, sample["query"]).model_dump()
        if task == "judge":
            return teacher.judge(image, sample["rule"]).model_dump()
        if task == "vqa":
            return teacher.vqa(image, sample["question"]).model_dump()
        if task == "compare":
            reference = (root / sample["reference_image"]).read_bytes()
            return teacher.compare(image, reference, sample["instruction"]).model_dump()
        raise ValueError(f"unknown task: {task}")

    return model
