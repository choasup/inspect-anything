"""冒烟训练后的结构化输出验证：取一条训练样本，用最新 checkpoint 推理，
确认输出能被 extract_json 解析且含期望字段。

用法：python3 scripts/check_smoke_output.py output/smoke
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from inspect_one.teacher.base import extract_json  # noqa: E402


def latest_checkpoint(root: str) -> str:
    ckpts = sorted(Path(root).glob("**/checkpoint-*"), key=lambda p: p.stat().st_mtime)
    if not ckpts:
        raise SystemExit(f"在 {root} 下找不到 checkpoint-*")
    return str(ckpts[-1])


def main() -> None:
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "output/smoke"
    ckpt = latest_checkpoint(out_dir)
    print(f"checkpoint: {ckpt}")

    sample = json.loads(Path("data/smoke/train_swift.jsonl").read_text("utf-8").splitlines()[0])
    user_msg = sample["messages"][0]["content"]
    image = sample["images"][0]

    from swift.llm import InferRequest, PtEngine, RequestConfig

    engine = PtEngine("Qwen/Qwen3-VL-2B-Instruct", adapters=[ckpt])
    req = InferRequest(
        messages=[{"role": "user", "content": user_msg}], images=[image]
    )
    resp = engine.infer([req], RequestConfig(max_tokens=512, temperature=0.0))
    text = resp[0].choices[0].message.content
    print(f"模型输出: {text}")

    parsed = extract_json(text)
    expected = json.loads(sample["messages"][1]["content"])
    missing = [k for k in expected if k not in parsed]
    if missing:
        raise SystemExit(f"FAILED: 输出缺少字段 {missing}（结构化协议未学会）")
    print(f"OK: 输出可解析，字段齐全 {list(parsed)}（真值: {expected}）")


if __name__ == "__main__":
    main()
