#!/usr/bin/env bash
# 冒烟训练：Qwen3-VL-2B + LoRA 过拟合合成数据，验证训练管线与结构化输出。
# 单卡：bash scripts/run_smoke.sh    四卡：NPROC_PER_NODE=4 bash scripts/run_smoke.sh
# 实验跟踪：REPORT_TO=swanlab bash scripts/run_smoke.sh（需先 swanlab login）
set -euo pipefail

test -f data/smoke/train_swift.jsonl || { echo "先运行 python3 scripts/make_smoke_data.py"; exit 1; }

swift sft \
  --model Qwen/Qwen3-VL-2B-Instruct \
  --dataset data/smoke/train_swift.jsonl \
  --train_type lora \
  --num_train_epochs 4 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 \
  --max_length 2048 \
  --logging_steps 5 \
  --save_strategy epoch \
  --report_to "${REPORT_TO:-tensorboard}" \
  --output_dir output/smoke

echo "== 训练完成，验证结构化输出 =="
python3 scripts/check_smoke_output.py output/smoke
