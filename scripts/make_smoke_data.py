"""生成自包含的冒烟训练数据：合成图像 + 精确标注。

灰底图上画 1-5 个彩色矩形，框坐标即真值——不依赖任何外部数据集，
专门用于验证「数据 → swift 训练 → 结构化输出」的管线是否通。
模型若连这个都学不会，说明训练配置有问题而不是数据有问题。

用法：python scripts/make_smoke_data.py [--n 64] [--out data/smoke]
产物：images/ + distilled.jsonl + train_swift.jsonl（已校验、可直接训练）
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw

import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from inspect_one.data.export_swift import export_jsonl  # noqa: E402
from inspect_one.data.format import Source, TrainingSample  # noqa: E402
from inspect_one.data.validate import validate_file  # noqa: E402
from inspect_one.schema import Task  # noqa: E402

PALETTE = {"红色": (220, 50, 50), "绿色": (50, 180, 80), "蓝色": (60, 90, 220), "黄色": (230, 200, 40)}
W, H = 640, 480


def make_image(rng: random.Random, path: Path) -> dict[str, list[dict]]:
    img = Image.new("RGB", (W, H), (128, 128, 128))
    draw = ImageDraw.Draw(img)
    boxes_by_color: dict[str, list[dict]] = {}
    placed: list[tuple[int, int, int, int]] = []
    for _ in range(rng.randint(1, 5)):
        color = rng.choice(list(PALETTE))
        # 重试找不与已有矩形重叠的位置，保证真值框无遮挡、绝对干净
        for _attempt in range(50):
            w, h = rng.randint(60, 150), rng.randint(60, 150)
            x = rng.randint(0, W - w - 1)
            y = rng.randint(0, H - h - 1)
            if all(
                x + w <= px or px + pw <= x or y + h <= py or py + ph <= y
                for px, py, pw, ph in placed
            ):
                break
        else:
            continue  # 放不下就少画一个
        placed.append((x, y, w, h))
        draw.rectangle([x, y, x + w, y + h], fill=PALETTE[color])
        boxes_by_color.setdefault(color, []).append(
            {"x1": x / W, "y1": y / H, "x2": (x + w) / W, "y2": (y + h) / H}
        )
    img.save(path, quality=92)
    return boxes_by_color


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--out", default="data/smoke")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    samples: list[TrainingSample] = []
    for i in range(args.n):
        rel = f"{args.out}/images/img_{i:04d}.jpg"
        boxes_by_color = make_image(rng, Path(rel))
        for color, boxes in boxes_by_color.items():
            query = f"{color}矩形"
            samples.append(
                TrainingSample(
                    sample_id=f"smoke-{i:04d}-{color}-ground",
                    image=rel,
                    task=Task.GROUND,
                    target={
                        "query": query,
                        "boxes": [{**b, "label": query, "score": 1.0} for b in boxes],
                    },
                    source=Source.DISTILLED,
                )
            )
            samples.append(
                TrainingSample(
                    sample_id=f"smoke-{i:04d}-{color}-count",
                    image=rel,
                    task=Task.COUNT,
                    target={
                        "query": query,
                        "count": len(boxes),
                        "points": [
                            {"x": (b["x1"] + b["x2"]) / 2, "y": (b["y1"] + b["y2"]) / 2}
                            for b in boxes
                        ],
                    },
                    source=Source.DISTILLED,
                )
            )
        has_red = "红色" in boxes_by_color
        samples.append(
            TrainingSample(
                sample_id=f"smoke-{i:04d}-judge",
                image=rel,
                task=Task.JUDGE,
                target={
                    "rule": "图中必须至少包含一个红色矩形",
                    "passed": has_red,
                    "confidence": 1.0,
                    "reason": "存在红色矩形" if has_red else "未发现红色矩形",
                    "evidence": [
                        {**b, "label": "红色矩形", "score": 1.0}
                        for b in boxes_by_color.get("红色", [])
                    ],
                },
                source=Source.DISTILLED,
            )
        )

    train_jsonl = out / "distilled.jsonl"
    with open(train_jsonl, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(s.model_dump_json() + "\n")
    errors = validate_file(train_jsonl)
    if errors:
        raise SystemExit("\n".join(errors))

    n = export_jsonl(samples, out / "train_swift.jsonl")
    print(f"OK: {args.n} images, {n} samples -> {out}/train_swift.jsonl")


if __name__ == "__main__":
    main()
