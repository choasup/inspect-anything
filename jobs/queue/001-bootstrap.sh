#!/usr/bin/env bash
# 任务 001：环境信息采集 + 依赖安装 + 自检 + 生成冒烟数据
set -euo pipefail

echo "===== system ====="
uname -a
echo "===== gpu ====="
rocm-smi --showproductname 2>/dev/null || amd-smi static 2>/dev/null | head -30 || echo "no rocm-smi/amd-smi"
rocm-smi 2>/dev/null | head -25 || true
echo "===== python/torch ====="
python3 --version
python3 -c "import torch; print('torch', torch.__version__, '| hip', torch.version.hip, '| gpus', torch.cuda.device_count())" 2>&1 || echo "torch not ready"
echo "===== disk/mem ====="
df -h / | tail -1
free -g | head -2

echo "===== bootstrap ====="
bash scripts/bootstrap_amd.sh

echo "===== smoke data ====="
python3 scripts/make_smoke_data.py
echo "===== 001 done ====="
