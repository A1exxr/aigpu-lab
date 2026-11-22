#!/usr/bin/env bash
# 在租来的 Hopper 机器上把环境一次配好，然后按顺序跑验证。
# 用法：
#   bash tools/setup_rented.sh              # 检查环境 -> 跑 4 步验证
#   bash tools/setup_rented.sh --setup-only # 只查/装环境，不跑 kernel
#   bash tools/setup_rented.sh --install    # 缺 torch 时用国内源装 torch cu124
set -u
cd "$(dirname "$0")/.." || exit 1
SETUP_ONLY=0
DO_INSTALL=0
for a in "$@"; do
  case "$a" in
    --setup-only) SETUP_ONLY=1 ;;
    --install)    DO_INSTALL=1 ;;
    *) echo "未知参数 $a"; exit 2 ;;
  esac
done

echo "=== 0) 硬件与驱动"
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv || exit 1
echo
echo "=== 1) 工具链"
command -v nvcc >/dev/null 2>&1 || { echo "找不到 nvcc：这个镜像没有 CUDA toolkit，换镜像或 conda install -c nvidia cuda-toolkit=12.4"; exit 1; }
nvcc --version | tail -2
command -v ninja >/dev/null 2>&1 || { echo "缺 ninja，正在装"; pip install -q ninja; }

echo
echo "=== 2) Python 环境"
python - <<'PY'
import sys
print("python", sys.version.split()[0])
try:
    import torch
except ImportError:
    print("NO_TORCH")
    sys.exit(3)
print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
if not torch.cuda.is_available():
    sys.exit(4)
cap = torch.cuda.get_device_capability(0)
print("compute_cap", cap)
sys.exit(0 if cap[0] * 10 + cap[1] >= 90 else 5)
PY
RC=$?
if [ $RC -eq 3 ] && [ $DO_INSTALL -eq 1 ]; then
  echo "装 torch（清华 PyPI，linux wheel 自带 cu12 runtime，不需要 CUDA toolkit 跑）..."
  pip install -q -i https://pypi.tuna.tsinghua.edu.cn/simple torch ninja || {
    echo "清华源失败，改用官方 PyPI"
    pip install -q -i https://pypi.org/simple torch ninja || exit 1
  }
  python -c "import torch;print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())" || exit 1
elif [ $RC -ne 0 ]; then
  echo "环境不对（rc=$RC）：torch 缺失或 compute_cap < 9.0。加 --install 可装 torch"
  exit 1
fi

echo
echo "=== 3) 仓库状态"
git rev-parse --short HEAD 2>/dev/null || echo "（不是 git 仓库，直接从本地跑也行）"

if [ $SETUP_ONLY -eq 1 ]; then
  echo
  echo "环境就绪（--setup-only，跳过 kernel）"
  exit 0
fi

echo
echo "=== 4) 验证链（判据见每步的「期望」）"
bash tools/run_on_hopper.sh
