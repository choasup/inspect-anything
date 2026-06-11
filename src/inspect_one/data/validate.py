"""数据文件自助校验 CLI。

用法：
  python -m inspect_one.data.validate data/distilled.jsonl          # TrainingSample 格式
  python -m inspect_one.data.validate bench/v0.1.jsonl --bench      # InspectBench 格式

逐行校验，汇报所有错误行（行号 + 原因）；全部通过退出码 0，否则 1。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from inspect_one.data.format import TrainingSample
from inspect_one.schema import Task

BENCH_SUBSETS = {"retail", "security", "industrial"}
BENCH_INPUT_KEYS = {
    Task.GROUND: ("query",),
    Task.COUNT: ("query",),
    Task.JUDGE: ("rule",),
    Task.VQA: ("question",),
    Task.COMPARE: ("instruction", "reference_image"),
}
BENCH_TARGET_KEYS = {
    Task.GROUND: ("boxes",),
    Task.COUNT: ("count",),
    Task.JUDGE: ("passed",),
    Task.VQA: ("answer",),
    Task.COMPARE: ("matches",),
}


def check_training(obj: dict) -> None:
    TrainingSample.model_validate(obj)


def check_bench(obj: dict) -> None:
    for key in ("id", "subset", "task", "image", "target"):
        if key not in obj:
            raise ValueError(f"missing field: {key}")
    if obj["subset"] not in BENCH_SUBSETS:
        raise ValueError(f"subset must be one of {sorted(BENCH_SUBSETS)}")
    task = Task(obj["task"])
    for key in BENCH_INPUT_KEYS[task]:
        if key not in obj:
            raise ValueError(f"task={task.value} requires field: {key}")
    for key in BENCH_TARGET_KEYS[task]:
        if key not in obj["target"]:
            raise ValueError(f"task={task.value} requires target.{key}")


def validate_file(path: str | Path, bench: bool = False) -> list[str]:
    errors: list[str] = []
    seen_ids: set[str] = set()
    id_key = "id" if bench else "sample_id"
    check = check_bench if bench else check_training
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                check(obj)
                sid = obj[id_key]
                if sid in seen_ids:
                    raise ValueError(f"duplicate {id_key}: {sid}")
                seen_ids.add(sid)
            except Exception as e:  # noqa: BLE001 - 汇总所有行错误统一汇报
                errors.append(f"line {lineno}: {e}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument("--bench", action="store_true", help="按 InspectBench 格式校验")
    args = parser.parse_args(argv)

    errors = validate_file(args.path, bench=args.bench)
    if errors:
        print(f"FAILED: {len(errors)} error(s) in {args.path}", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        return 1
    n = sum(1 for l in open(args.path, encoding="utf-8") if l.strip())
    print(f"OK: {n} samples in {args.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
