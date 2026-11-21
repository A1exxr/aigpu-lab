#!/usr/bin/env bash
# 在租来的 Hopper 机器上按顺序验三件事：环境 -> wgmma 语义 -> w12e -> w12f
# 用法：bash run_on_hopper.sh            （假定已 git clone 本仓库并装好 torch）
set -u
cd "$(dirname "$0")/.." || exit 1
echo "=== 0) 环境"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    print("compute_cap", cap)
    assert cap[0] * 10 + cap[1] >= 90, "必须是 Hopper（sm_90a）"
PY

echo
echo "=== 1) wgmma 描述符语义（3 分钟，错了后面两个都会错）"
python kernels/wgmma_warmup.py || { echo "warmup 失败 -> 描述符/LBO/SBO 有问题，先修这里"; exit 1; }

echo
echo "=== 2) w12e（QK=wgmma，K/V 走 cp.async）"
python kernels/w12e_fa2_wgmma.py
echo "期望：3.644e-04 / 4.944e-04 / 5.429e-04"

echo
echo "=== 3) w12f（K/V 也走 TMA）"
python kernels/w12f_fa2_tma.py
echo "期望：与上面三档位级一致"

echo
echo "=== 4) w12g（warp specialization：producer 只发 TMA / consumer 只算）"
python kernels/w12g_fa2_ws.py
echo "期望：与上面三档位级一致"
