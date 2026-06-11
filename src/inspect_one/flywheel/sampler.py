"""价值采样器：飞轮的核心。给每条线上推理记录打「训练价值分」，
分层配额下取 top-k 送标注，而不是全量标。

六路信号对应 docs/03-data-flywheel.md 第 3 节：
低置信度 / 师生分歧 / 双头分歧 / 用户改判 / 分布新颖度 / 时间衰减。
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class InferenceRecord:
    record_id: str
    scene: str  # 分层配额的键，如 "retail/客户A" / "industrial/产线3"
    confidence: float  # 模型对本次输出的置信度 [0,1]
    teacher_disagreement: float = 0.0  # 与教师抽检输出的差异度 [0,1]
    head_disagreement: float = 0.0  # det 头与 LLM 头结论冲突度 [0,1]
    user_override: bool = False  # 人工改判（巡店员/值班员/质检员推翻模型）
    novelty: float = 0.0  # 视觉嵌入到训练集聚类的归一化距离 [0,1]
    age_days: float = 0.0  # 样本距今天数，用于时间衰减


@dataclass
class SamplerConfig:
    w_uncertainty: float = 1.0
    w_teacher_disagreement: float = 2.0
    w_head_disagreement: float = 1.5
    w_user_override: float = 4.0  # 确定性错误，价值最高
    w_novelty: float = 1.5
    decay_halflife_days: float = 30.0  # 时间衰减半衰期
    per_scene_quota: int = 200  # 单场景配额上限，防大客户淹没采样池
    quotas: dict[str, int] = field(default_factory=dict)  # 按场景覆盖默认配额


def value_score(r: InferenceRecord, cfg: SamplerConfig) -> float:
    raw = (
        cfg.w_uncertainty * (1.0 - r.confidence)
        + cfg.w_teacher_disagreement * r.teacher_disagreement
        + cfg.w_head_disagreement * r.head_disagreement
        + cfg.w_user_override * (1.0 if r.user_override else 0.0)
        + cfg.w_novelty * r.novelty
    )
    decay = math.exp(-math.log(2.0) * r.age_days / cfg.decay_halflife_days)
    return raw * decay


def select(
    records: list[InferenceRecord], budget: int, cfg: SamplerConfig | None = None
) -> list[InferenceRecord]:
    """全局预算 + 场景配额下选 top 价值样本。

    先在每个场景内部按价值分排序并截断到该场景配额，
    再全局排序取预算内的 top-k，保证返回顺序为价值降序。
    """
    cfg = cfg or SamplerConfig()
    by_scene: dict[str, list[tuple[float, InferenceRecord]]] = defaultdict(list)
    for r in records:
        by_scene[r.scene].append((value_score(r, cfg), r))

    pooled: list[tuple[float, InferenceRecord]] = []
    for scene, scored in by_scene.items():
        scored.sort(key=lambda t: t[0], reverse=True)
        quota = cfg.quotas.get(scene, cfg.per_scene_quota)
        pooled.extend(scored[:quota])

    pooled.sort(key=lambda t: t[0], reverse=True)
    return [r for _, r in pooled[:budget]]
