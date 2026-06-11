"""Qwen3-VL 教师客户端（OpenAI 兼容接口，适配 vLLM 自托管与云 API）。

只依赖标准库 urllib，方便在标注集群的最小镜像里运行。
prompt 模板强制 JSON 输出，坐标为 [0,1] 归一化 xyxy，与 schema 对齐。
"""

from __future__ import annotations

import base64
import json
import urllib.request

from inspect_one.schema import (
    Box,
    CompareDiff,
    CompareResult,
    CountResult,
    GroundResult,
    JudgeResult,
    Point,
    ScoredBox,
    VQAResult,
)
from inspect_one.teacher.base import Teacher, extract_json

_COORD_SPEC = (
    "Coordinates are normalized to [0,1] relative to image width/height, "
    "format xyxy. Respond with JSON only, no prose."
)

PROMPTS = {
    "ground": (
        "Locate every instance of: {query}. "
        '{spec} Schema: {{"boxes": [{{"x1":f,"y1":f,"x2":f,"y2":f,"label":s,"score":f}}]}}'
    ),
    "count": (
        "Count every instance of: {query}. Mark each with a center point. "
        '{spec} Schema: {{"count":i,"points":[{{"x":f,"y":f}}]}}'
    ),
    "compare": (
        "First image is the actual scene, second is the reference. {instruction} "
        '{spec} Schema: {{"matches":b,"diffs":[{{"region":{{"x1":f,"y1":f,"x2":f,"y2":f}},'
        '"expected":s,"actual":s}}]}}'
    ),
    "vqa": (
        "{question} "
        '{spec} Schema: {{"answer":s,"region":{{"x1":f,"y1":f,"x2":f,"y2":f}} or null}}'
    ),
    "judge": (
        "Audit this image against the rule: {rule}. "
        '{spec} Schema: {{"passed":b,"confidence":f,"reason":s,'
        '"evidence":[{{"x1":f,"y1":f,"x2":f,"y2":f,"label":s,"score":f}}]}}'
    ),
}


class QwenVLTeacher(Teacher):
    def __init__(
        self,
        base_url: str,
        model: str = "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8",
        api_key: str = "EMPTY",
        timeout: float = 120.0,
        temperature: float = 0.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.temperature = temperature

    # -- transport -------------------------------------------------------

    def _chat(self, prompt: str, images: list[bytes]) -> str:
        content: list[dict] = [
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/jpeg;base64,"
                    + base64.b64encode(img).decode("ascii")
                },
            }
            for img in images
        ]
        content.append({"type": "text", "text": prompt})
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": content}],
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read())
        return body["choices"][0]["message"]["content"]

    # -- primitives ------------------------------------------------------

    def ground(self, image: bytes, query: str) -> GroundResult:
        raw = extract_json(
            self._chat(PROMPTS["ground"].format(query=query, spec=_COORD_SPEC), [image])
        )
        return GroundResult(
            query=query, boxes=[ScoredBox(**b) for b in raw.get("boxes", [])]
        )

    def count(self, image: bytes, query: str) -> CountResult:
        raw = extract_json(
            self._chat(PROMPTS["count"].format(query=query, spec=_COORD_SPEC), [image])
        )
        return CountResult(
            query=query,
            count=int(raw["count"]),
            points=[Point(**p) for p in raw.get("points", [])],
        )

    def compare(self, image: bytes, reference: bytes, instruction: str) -> CompareResult:
        raw = extract_json(
            self._chat(
                PROMPTS["compare"].format(instruction=instruction, spec=_COORD_SPEC),
                [image, reference],
            )
        )
        return CompareResult(
            matches=bool(raw["matches"]),
            diffs=[
                CompareDiff(
                    region=Box(**d["region"]),
                    expected=d["expected"],
                    actual=d["actual"],
                )
                for d in raw.get("diffs", [])
            ],
        )

    def vqa(self, image: bytes, question: str) -> VQAResult:
        raw = extract_json(
            self._chat(PROMPTS["vqa"].format(question=question, spec=_COORD_SPEC), [image])
        )
        region = raw.get("region")
        return VQAResult(
            question=question,
            answer=str(raw["answer"]),
            region=Box(**region) if region else None,
        )

    def judge(self, image: bytes, rule: str) -> JudgeResult:
        raw = extract_json(
            self._chat(PROMPTS["judge"].format(rule=rule, spec=_COORD_SPEC), [image])
        )
        return JudgeResult(
            rule=rule,
            passed=bool(raw["passed"]),
            confidence=float(raw.get("confidence", 1.0)),
            reason=str(raw.get("reason", "")),
            evidence=[ScoredBox(**b) for b in raw.get("evidence", [])],
        )
