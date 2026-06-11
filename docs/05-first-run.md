# 05 · 首轮训练 Runbook（学生 v0）

> 目标：用「公开数据 + 既有离线蒸馏数据」微调 Qwen3-VL 小档位得到学生 v0，
> 并产出第一份 InspectBench 报告（学生 vs 教师零样本基线）。
> 本仓库代码全程无 GPU 依赖；训练与推理在你的 GPU 机器上由 ms-swift 承担。

## 0. 环境（GPU 机器）

```bash
git clone <repo> && cd inspect-anything
pip install -e ".[dev]" && pytest          # 36 个测试应全绿
pip install ms-swift swanlab               # 训练框架 + 实验跟踪
```

## 1. 数据准备

### 1.1 蒸馏数据 → TrainingSample JSONL

把每条线下蒸馏样本转成 `TrainingSample`（字段见 `data/format.py`），`source` 填
`distilled`，监督目标按任务原语组织（坐标归一化 xyxy）。一行一条存为
`data/distilled.jsonl`。格式不确定的话，把脱敏样例发出来，转换脚本由仓库统一提供。

### 1.2 公开数据 → TrainingSample

COCO 风格标注（COCO/Objects365/LVIS）直接转，自动派生 ground + count 两类样本：

```python
import json
from inspect_one.data.converters import coco_to_samples

public = coco_to_samples(json.load(open("annotations/instances_train2017.json")))
```

### 1.3 混合并导出 ms-swift 格式

```python
from inspect_one.data.format import Source, TrainingSample
from inspect_one.data.mixture import MixtureSpec, mix
from inspect_one.data.export_swift import export_jsonl

distilled = [TrainingSample.model_validate_json(l)
             for l in open("data/distilled.jsonl", encoding="utf-8")]

# 首轮无金标，30/50/20 自动重归一为 3:5
epoch = mix({Source.PUBLIC: public, Source.DISTILLED: distilled},
            epoch_size=200_000, spec=MixtureSpec(seed=0))
export_jsonl(epoch, "data/train_swift.jsonl")
```

记录数据版本：`distilled.jsonl` 的哈希 + 公开数据集版本 + seed，写进实验名。

## 2. 冒烟训练（先小后大，必做）

取 500 条先过拟合一遍，验证三件事：流程能跑通、loss 能下降、
**模型输出能被 `extract_json` 解析**（结构化协议是否学到位，这是最常见的翻车点）：

```bash
head -500 data/train_swift.jsonl > data/smoke.jsonl
swift sft \
  --model Qwen/Qwen3-VL-2B-Instruct \
  --dataset data/smoke.jsonl \
  --train_type lora \
  --num_train_epochs 5 \
  --report_to swanlab \
  --output_dir output/smoke
```

冒烟通过后再上全量（同命令换 `--dataset data/train_swift.jsonl`、epoch 1-2）。
不想写命令行就 `swift web-ui` 在网页里配。

## 3. 评测：学生 vs 教师零样本

### 3.1 InspectBench v0.1 种子集（与训练并行准备）

每场景先 50-100 条，JSONL 格式见 `bench/runner.py` 文档串。来源：业务真实图片
人工标注，或留出一部分蒸馏数据（**留出样本的 id 必须从训练集排除**）。

### 3.2 部署学生并跑评测

```bash
swift deploy --model output/v0/checkpoint-xxx --port 8000   # OpenAI 兼容服务
```

```python
from inspect_one.bench.adapters import teacher_as_model
from inspect_one.bench.runner import evaluate, load_jsonl, regression_gate
from inspect_one.teacher.qwen_vl import QwenVLTeacher

samples = load_jsonl("bench/v0.1.jsonl")
student = teacher_as_model(
    QwenVLTeacher("http://localhost:8000/v1", model="default"), image_root="bench/images")
teacher = teacher_as_model(
    QwenVLTeacher("<教师 endpoint>/v1", model="Qwen3-VL-30B-A3B"), image_root="bench/images")

student_report = evaluate(samples, student)
teacher_report = evaluate(samples, teacher)   # 这份就是零样本基线，存档
print(student_report)
print(regression_gate(student_report, teacher_report, max_drop=0.05))
```

教师报告是后续所有迭代的固定基线（SwanLab 里 pin 住）；Phase 1 出口判据 =
学生达到教师的 95%。

## 4. 完成后的状态

- [ ] 学生 v0 checkpoint + 训练曲线（SwanLab）
- [ ] InspectBench v0.1 种子集（评测样本与训练集 id 级隔离）
- [ ] 教师零样本基线报告（存档为基线）
- [ ] 学生 vs 教师对比报告
- 下一步进入飞轮：部署埋点 → 价值采样 → 增量标注 →（数据量到阈值）重训 v1
