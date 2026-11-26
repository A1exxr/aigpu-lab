"""w13_fa3.cu 的 driver：三档 shape 对 SDPA，打印 max|diff| 与 TFLOPS。

用法（WSL）：
    export PATH=$HOME/miniconda3/envs/triton/bin:/usr/local/cuda/bin:$PATH
    python kernels/fa3.py

只做正确性 + 粗略吞吐；需要 Hopper（compute_cap == 9.0），否则直接退出。
"""
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

HERE = Path(__file__).resolve().parent
CC_MAJOR, CC_MINOR = torch.cuda.get_device_capability()
if CC_MAJOR != 9:
    print(f"[skip] 需要 Hopper（sm_90a），当前是 sm_{CC_MAJOR}{CC_MINOR}")
    sys.exit(0)

ARCH = f"-arch=sm_{CC_MAJOR}{CC_MINOR}a"
SRC = (HERE / "w13_fa3.cu").read_text(encoding="utf-8")

mod = load_inline(
    name="fa3_ext",
    cpp_sources="",
    cuda_sources=SRC,
    functions=["fa3_forward"],
    extra_cuda_cflags=["-O3", "-lineinfo", ARCH, "-Xptxas", "-v"],
    verbose=os.environ.get("FA3_VERBOSE") == "1",
)

# 三个 shape：(B, H, Hkv, S, D)
CASES = [(2, 8, 2, 512, 64), (2, 8, 2, 125, 64), (2, 8, 2, 1024, 64)]
# 前两版（wgmma / TMA）在同一批 shape 上的 max|diff|，用来判断是否位级一致
EXPECT = [3.644e-04, 4.944e-04, 5.429e-04]


def run(B, H, Hkv, S, D, iters=30):
    torch.manual_seed(0)
    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.half)
    k = torch.randn(B, Hkv, S, D, device="cuda", dtype=torch.half)
    v = torch.randn(B, Hkv, S, D, device="cuda", dtype=torch.half)
    scale = 1.0 / (D ** 0.5)

    ref = F.scaled_dot_product_attention(
        q, k.expand(B, Hkv, S, D).repeat_interleave(H // Hkv, dim=1),
        v.expand(B, Hkv, S, D).repeat_interleave(H // Hkv, dim=1),
        is_causal=True, scale=scale,
    )
    out = mod.fa3_forward(q, k, v, scale)
    diff = (out.float() - ref.float()).abs().max().item()

    for _ in range(10):
        mod.fa3_forward(q, k, v, scale)
    torch.cuda.synchronize()
    ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
    ev0.record()
    for _ in range(iters):
        mod.fa3_forward(q, k, v, scale)
    ev1.record()
    torch.cuda.synchronize()
    ms = ev0.elapsed_time(ev1) / iters

    flops = 2 * B * H * S * (S + 1) * D        # causal 要减半
    return diff, ms, flops / ms / 1e9


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}  sm_{CC_MAJOR}{CC_MINOR}  arch={ARCH}")
    ok = True
    for (B, H, Hkv, S, D), exp in zip(CASES, EXPECT):
        diff, ms, tflops = run(B, H, Hkv, S, D)
        bit = "位级一致" if abs(diff - exp) < 1e-9 else f"（基准 {exp:.3e}）"
        passed = diff <= 3e-3
        ok &= passed
        print(f"B={B} H={H} Hkv={Hkv} S={S:5d}  max|diff|={diff:.3e} {bit}  "
              f"{ms:.3f} ms  {tflops:6.2f} TFLOPS  {'PASS' if passed else 'FAIL'}")
    print("结论：" + ("全部 PASS" if ok else "有 FAIL，先看 diff 哪一档开始偏"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
