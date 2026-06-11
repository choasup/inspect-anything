#!/usr/bin/env bash
# 任务 002：冒烟训练（Qwen3-VL-2B + LoRA，单卡）+ 结构化输出验证
set -euo pipefail

bash scripts/run_smoke.sh
echo "===== 002 done ====="
