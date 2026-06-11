"""训练数据混合器：按来源比例组装一个 epoch 的样本清单。

实现 docs/02 训练配方的混配（默认 30% 公开 / 50% 蒸馏 / 20% 金标）。
同一 seed 产出完全相同的清单——数据版本 + seed 即可复现任意一轮训练。
来源样本不足配额时有放回重复采样（小金标集天然需要过采样）。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from inspect_one.data.format import Source, TrainingSample

DEFAULT_RATIOS: dict[Source, float] = {
    Source.PUBLIC: 0.30,
    Source.DISTILLED: 0.50,
    Source.GOLD: 0.20,
}


@dataclass
class MixtureSpec:
    ratios: dict[Source, float] = field(default_factory=lambda: dict(DEFAULT_RATIOS))
    seed: int = 0

    def __post_init__(self) -> None:
        total = sum(self.ratios.values())
        if total <= 0:
            raise ValueError("ratios must sum to a positive value")
        self.ratios = {k: v / total for k, v in self.ratios.items()}


def mix(
    datasets: dict[Source, list[TrainingSample]],
    epoch_size: int,
    spec: MixtureSpec | None = None,
) -> list[TrainingSample]:
    spec = spec or MixtureSpec()
    rng = random.Random(spec.seed)

    epoch: list[TrainingSample] = []
    allocated = 0
    sources = [s for s in spec.ratios if datasets.get(s)]
    if not sources:
        raise ValueError("no non-empty dataset matches the mixture ratios")
    live_total = sum(spec.ratios[s] for s in sources)

    for i, source in enumerate(sources):
        if i == len(sources) - 1:
            quota = epoch_size - allocated  # 余数全给最后一个来源，保证总量精确
        else:
            quota = round(epoch_size * spec.ratios[source] / live_total)
        allocated += quota
        pool = datasets[source]
        if len(pool) >= quota:
            epoch.extend(rng.sample(pool, quota))
        else:
            epoch.extend(pool)
            epoch.extend(rng.choices(pool, k=quota - len(pool)))  # 过采样补齐

    rng.shuffle(epoch)
    return epoch
