from inspect_one.bench.adapters import teacher_as_model
from inspect_one.schema import (
    CompareResult,
    CountResult,
    GroundResult,
    JudgeResult,
    ScoredBox,
    VQAResult,
)
from inspect_one.teacher.base import Teacher


class FakeTeacher(Teacher):
    def ground(self, image, query):
        assert image == b"img-bytes"
        return GroundResult(
            query=query,
            boxes=[ScoredBox(x1=0.1, y1=0.1, x2=0.3, y2=0.3, label=query, score=0.9)],
        )

    def count(self, image, query):
        return CountResult(query=query, count=4)

    def compare(self, image, reference, instruction):
        assert reference == b"ref-bytes"
        return CompareResult(matches=True)

    def vqa(self, image, question):
        return VQAResult(question=question, answer="是")

    def judge(self, image, rule):
        return JudgeResult(rule=rule, passed=True)


def test_dispatch_by_task(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"img-bytes")
    (tmp_path / "ref.jpg").write_bytes(b"ref-bytes")
    model = teacher_as_model(FakeTeacher(), image_root=tmp_path)

    out = model({"task": "ground", "image": "a.jpg", "query": "可乐"})
    assert out["boxes"][0]["label"] == "可乐"

    out = model({"task": "count", "image": "a.jpg", "query": "瓶装水"})
    assert out["count"] == 4

    out = model({"task": "judge", "image": "a.jpg", "rule": "必须戴安全帽"})
    assert out["passed"] is True

    out = model(
        {"task": "compare", "image": "a.jpg", "reference_image": "ref.jpg",
         "instruction": "对比陈列"}
    )
    assert out["matches"] is True
