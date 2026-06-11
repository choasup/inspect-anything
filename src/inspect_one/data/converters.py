"""公开数据集 → TrainingSample 转换器。

先实现 COCO 格式（Objects365/LVIS 等均提供 COCO 风格标注），
同一份检测标注可同时派生 ground 和 count 两类原语样本。
"""

from __future__ import annotations

from inspect_one.data.format import Source, TrainingSample
from inspect_one.schema import Task


def _normalize_bbox(bbox: list[float], width: float, height: float) -> dict:
    """COCO xywh 像素坐标 → 归一化 xyxy，越界部分裁剪到图像内。"""
    x, y, w, h = bbox
    return {
        "x1": max(0.0, min(1.0, x / width)),
        "y1": max(0.0, min(1.0, y / height)),
        "x2": max(0.0, min(1.0, (x + w) / width)),
        "y2": max(0.0, min(1.0, (y + h) / height)),
    }


def coco_to_samples(
    coco: dict,
    source: Source = Source.PUBLIC,
    derive_count: bool = True,
    min_boxes: int = 1,
) -> list[TrainingSample]:
    """COCO 标注字典 → ground（每图每类别一条）与 count 样本。

    coco 需含 images / annotations / categories 三个标准字段。
    iscrowd 标注跳过；同图同类别的框聚合进一条 ground 样本。
    """
    categories = {c["id"]: c["name"] for c in coco["categories"]}
    images = {i["id"]: i for i in coco["images"]}

    grouped: dict[tuple[int, int], list[dict]] = {}
    for ann in coco["annotations"]:
        if ann.get("iscrowd"):
            continue
        img = images.get(ann["image_id"])
        if img is None:
            continue
        key = (ann["image_id"], ann["category_id"])
        grouped.setdefault(key, []).append(
            _normalize_bbox(ann["bbox"], img["width"], img["height"])
        )

    samples: list[TrainingSample] = []
    for (image_id, category_id), boxes in grouped.items():
        if len(boxes) < min_boxes:
            continue
        img = images[image_id]
        label = categories[category_id]
        samples.append(
            TrainingSample(
                sample_id=f"{image_id}-{category_id}-ground",
                image=img["file_name"],
                task=Task.GROUND,
                target={
                    "query": label,
                    "boxes": [{**b, "label": label, "score": 1.0} for b in boxes],
                },
                source=source,
            )
        )
        if derive_count:
            samples.append(
                TrainingSample(
                    sample_id=f"{image_id}-{category_id}-count",
                    image=img["file_name"],
                    task=Task.COUNT,
                    target={
                        "query": label,
                        "count": len(boxes),
                        "points": [
                            {"x": (b["x1"] + b["x2"]) / 2, "y": (b["y1"] + b["y2"]) / 2}
                            for b in boxes
                        ],
                    },
                    source=source,
                )
            )
    return samples
