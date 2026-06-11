import pytest

from inspect_one.flywheel.sampler import (
    InferenceRecord,
    SamplerConfig,
    select,
    value_score,
)


def rec(record_id, scene="retail/a", **kw):
    defaults = dict(confidence=0.9)
    defaults.update(kw)
    return InferenceRecord(record_id=record_id, scene=scene, **defaults)


def test_user_override_dominates():
    cfg = SamplerConfig()
    override = value_score(rec("a", user_override=True), cfg)
    uncertain = value_score(rec("b", confidence=0.0), cfg)
    assert override > uncertain


def test_time_decay_halves_score():
    cfg = SamplerConfig(decay_halflife_days=30.0)
    fresh = value_score(rec("a", confidence=0.0), cfg)
    old = value_score(rec("b", confidence=0.0, age_days=30.0), cfg)
    assert old == pytest.approx(fresh / 2)


def test_scene_quota_prevents_flooding():
    # 大客户 100 条高分样本，小客户 5 条低分样本；配额=3 时小客户仍有席位
    records = [
        rec(f"big{i}", scene="retail/big", confidence=0.0) for i in range(100)
    ] + [rec(f"small{i}", scene="retail/small", confidence=0.5) for i in range(5)]
    picked = select(records, budget=8, cfg=SamplerConfig(per_scene_quota=3))
    scenes = {r.scene for r in picked}
    assert "retail/small" in scenes
    assert sum(1 for r in picked if r.scene == "retail/big") == 3


def test_select_orders_by_value_and_respects_budget():
    records = [
        rec("low", confidence=0.9),
        rec("high", confidence=0.1),
        rec("override", confidence=0.9, user_override=True),
    ]
    picked = select(records, budget=2)
    assert [r.record_id for r in picked] == ["override", "high"]
