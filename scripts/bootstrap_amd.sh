#!/usr/bin/env bash
# AMD(ROCm) 机器一键环境准备。在仓库根目录执行：bash scripts/bootstrap_amd.sh
set -euo pipefail

echo "== GPU =="
rocm-smi --showproductname 2>/dev/null || amd-smi static 2>/dev/null | head -20 \
  || echo "WARN: rocm-smi/amd-smi 不可用，请确认 ROCm 驱动已装"

echo "== PyTorch =="
if python3 -c "import torch; assert torch.cuda.is_available() and torch.version.hip" 2>/dev/null; then
  python3 -c "import torch; print('torch', torch.__version__, '| hip', torch.version.hip, '| gpus', torch.cuda.device_count())"
else
  echo "未检测到可用的 ROCm 版 torch，安装中（ROCm 6.4 wheel；驱动版本不同请改 index-url）..."
  pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4
  python3 -c "import torch; print('torch', torch.__version__, '| hip', torch.version.hip, '| gpus', torch.cuda.device_count())"
fi

echo "== 项目依赖 =="
pip install -e ".[dev]"
pip install ms-swift swanlab pillow

echo "== 自检 =="
python3 -m pytest -q

echo "== 完成 =="
echo "下一步："
echo "  python3 scripts/make_smoke_data.py        # 生成冒烟数据"
echo "  bash scripts/run_smoke.sh                 # 单卡冒烟训练"
echo "  NPROC_PER_NODE=4 bash scripts/run_smoke.sh  # 4 卡"
