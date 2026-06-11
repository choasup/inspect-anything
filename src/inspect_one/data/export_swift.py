"""TrainingSample → ms-swift SFT 数据格式（messages + images JSONL）。

设计原则：学生的训练目标格式 == 我们的推理输出格式。
user prompt 复用教师层的同一套模板（teacher/qwen_vl.py 的 PROMPTS），
assistant 回复为目标的紧凑 JSON——学生从第一天就说我们的结构化协议，
推理侧的 extract_json 解析路径对教师和学生通用。
坐标保持归一化 [0,1]，不依赖图像尺寸，导出时无需读图。
"""

from __future__ import annotations

import json
from pathlib import Path

from inspect_one.data.format import TrainingSample
from inspect_one.schema import Task
from inspect_one.teacher.qwen_vl import _COORD_SPEC, PROMPTS

_TARGET_KEYS = {
    Task.GROUND: ("boxes",),
    Task.COUNT: ("count", "points"),
    Task.COMPARE: ("matches", "diffs"),
    Task.VQA: ("answer", "region"),
    Task.JUDGE: ("passed", "confidence", "reason", "evidence"),
}


def _prompt(sample: TrainingSample) -> str:
    t = sample.task
    if t in (Task.GROUND, Task.COUNT):
        return PROMPTS[t.value].format(query=sample.target["query"], spec=_COORD_SPEC)
    if t == Task.VQA:
        return PROMPTS["vqa"].format(question=sample.target["question"], spec=_COORD_SPEC)
    if t == Task.JUDGE:
        return PROMPTS["judge"].format(rule=sample.target["rule"], spec=_COORD_SPEC)
    return PROMPTS["compare"].format(
        instruction=sample.target.get("instruction", "Compare the two images."),
        spec=_COORD_SPEC,
    )


def to_swift(sample: TrainingSample) -> dict:
    answer = {k: sample.target[k] for k in _TARGET_KEYS[sample.task] if k in sample.target}
    images = [sample.image]
    if sample.task == Task.COMPARE and sample.target.get("reference_image"):
        images.append(sample.target["reference_image"])
    return {
        "messages": [
            {"role": "user", "content": "<image>" * len(images) + _prompt(sample)},
            {
                "role": "assistant",
                "content": json.dumps(answer, ensure_ascii=False, separators=(",", ":")),
            },
        ],
        "images": images,
    }


def export_jsonl(samples: list[TrainingSample], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(to_swift(s), ensure_ascii=False) + "\n")
    return len(samples)
