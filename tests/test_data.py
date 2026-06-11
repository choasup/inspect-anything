from collections import Counter

import pytest
from pydantic import ValidationError

from inspect_one.data.converters import coco_to_samples
from inspect_one.data.format import Source, TrainingSample
from inspect_one.data.mixture import MixtureSpec, mix
from inspect_one.schema import Task

COCO = {
    "images": [{"id": 1, "file_name": "shelf.jpg", "width": 1000, "height": 500}],
    "categories": [{"id": 7, "name": "可乐 330ml"}],
    "annotations": [
        {"id": 1, "image_id": 1, "category_id": 7, "bbox": [100, 50, 200, 100]},
        {"id": 2, "image_id": 1, "category_id": 7, "bbox": [400, 50, 200, 100]},
        {"id": 3, "image_id": 1, "category_id": 7, "bbox": [0, 0, 10, 10], "iscrowd": 1},
    ],
}


def make(sample_id, source, task=Task.COUNT):
    return TrainingSample(
        sample_id=sample_id,
        image="x.jpg",
        task=task,
        target={"query": "q", "count": 1, "points": []},
        source=source,
    )


def test_target_must_match_task():
    with pytest.raises(ValidationError):
        TrainingSample(
            sample_id="a",
            image="x.jpg",
            task=Task.GROUND,
            target={"count": 3},  # count 负载配 ground 任务
            source=Source.PUBLIC,
        )


def test_coco_conversion_normalizes_and_skips_crowd():
    samples = coco_to_samples(COCO)
    ground = next(s for s in samples if s.task == Task.GROUND)
    count = next(s for s in samples if s.task == Task.COUNT)
    assert len(ground.target["boxes"]) == 2  # iscrowd 被跳过
    b = ground.target["boxes"][0]
    assert (b["x1"], b["y1"], b["x2"], b["y2"]) == (0.1, 0.1, 0.3, 0.3)
    assert count.target["count"] == 2
    assert count.target["points"][0] == {"x": 0.2, "y": 0.2}


def test_mixture_ratios_and_oversampling():
    datasets = {
        Source.PUBLIC: [make(f"p{i}", Source.PUBLIC) for i in range(1000)],
        Source.DISTILLED: [make(f"d{i}", Source.DISTILLED) for i in range(1000)],
        Source.GOLD: [make(f"g{i}", Source.GOLD) for i in range(10)],  # 金标少，需过采样
    }
    epoch = mix(datasets, epoch_size=100)
    counts = Counter(s.source for s in epoch)
    assert len(epoch) == 100
    assert counts[Source.PUBLIC] == 30
    assert counts[Source.DISTILLED] == 50
    assert counts[Source.GOLD] == 20  # 10 条全量 + 10 条重复


def test_mixture_deterministic_by_seed():
    datasets = {Source.PUBLIC: [make(f"p{i}", Source.PUBLIC) for i in range(50)]}
    spec = MixtureSpec(ratios={Source.PUBLIC: 1.0}, seed=42)
    a = [s.sample_id for s in mix(datasets, 20, spec)]
    b = [s.sample_id for s in mix(datasets, 20, MixtureSpec(ratios={Source.PUBLIC: 1.0}, seed=42))]
    assert a == b


def test_mixture_renormalizes_when_source_missing():
    # 没有金标时（首轮训练），其余来源按比例重归一，总量不变
    datasets = {
        Source.PUBLIC: [make(f"p{i}", Source.PUBLIC) for i in range(1000)],
        Source.DISTILLED: [make(f"d{i}", Source.DISTILLED) for i in range(1000)],
    }
    epoch = mix(datasets, epoch_size=80)
    counts = Counter(s.source for s in epoch)
    assert len(epoch) == 80
    assert counts[Source.PUBLIC] == 30  # 0.3/0.8
    assert counts[Source.DISTILLED] == 50  # 0.5/0.8
