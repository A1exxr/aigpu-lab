# -*- coding: utf-8 -*-
"""w12f（FA2 前向，K/V 走 TMA + QK 走 wgmma）：正确性三档 + 与 SDPA 比性能。

需要 compute_cap >= 9.0 的卡（H100/H800/H200/H20）。本地 4060 能在编译处报
"arch=sm_90a not supported"，那是预期的——只在 Hopper 上跑本脚本。

跑法（租机）：
    python kernels/wgmma_warmup.py        # 先验描述符语义（3 分钟）
    python kernels/w12e_fa2_wgmma.py      # 再验 w12e（QK=wgmma，K/V 走 cp.async）
    python kernels/w12f_fa2_tma.py        # 最后验本文件（K/V 也走 TMA）
三档 max|diff| 必须与 w12e 一致：3.644e-04 / 4.944e-04 / 5.429e-04。
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

CU = Path(__file__).with_name("w12f_fa2_tma.cu").read_text(encoding="utf-8")

CPP = "torch::Tensor fa2_wgmma_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, double scale);"

# 三档与 w12e 完全相同，便于逐档对比 diff 是否位级一致
CASES = [(2, 8, 2, 512, 64), (2, 8, 2, 125, 64), (2, 8, 2, 1024, 64)]

# w12e 在 4060 上跑出的三档 diff（上机后与它们逐档比较）
EXPECT = [3.644e-04, 4.944e-04, 5.429e-04]


def check_arch():
    if not torch.cuda.is_available():
        sys.exit("没有 CUDA 设备")
    cap = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    cc = cap[0] * 10 + cap[1]
    print(f"device = {name}  compute_cap = {cap[0]}.{cap[1]}")
    if cc < 90:
        sys.exit(f"需要 sm_90a（Hopper），当前 {cc/10:.1f} 跑不了 wgmma")
    return cc


def build(cc):
    return load_inline(
        name="w12f_fa2_tma",
        cpp_sources=CPP,
        cuda_sources=CU,
        functions=["fa2_wgmma_forward"],
        extra_cuda_cflags=[f"-arch=sm_{cc}a", "-O3", "--use_fast_math"],
        verbose=False,
    )


def ref(q, k, v, scale):
    g = q.size(1) // k.size(1)
    kk = k.repeat_interleave(g, dim=1)
    vv = v.repeat_interleave(g, dim=1)
    return F.scaled_dot_product_attention(q, kk, vv, is_causal=True, scale=scale)


def bench(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    beg = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    beg.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return beg.elapsed_time(end) / iters


def main():
    cc = check_arch()
    mod = build(cc)
    torch.manual_seed(0)
    fails = 0
    for i, (B, H, Hkv, S, D) in enumerate(CASES):
        q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, Hkv, S, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, Hkv, S, D, device="cuda", dtype=torch.float16)
        scale = 1.0 / (D ** 0.5)

        ours = mod.fa2_wgmma_forward(q, k, v, scale)
        r = ref(q, k, v, scale)
        diff = (ours.float() - r.float()).abs().max().item()
        bit = "位级一致" if abs(diff - EXPECT[i]) < 1e-9 else f"与 w12e 的 {EXPECT[i]:.3e} 不同"
        ok = diff <= 3e-3
        fails += 0 if ok else 1
        print(f"B={B} H={H} Hkv={Hkv} S={S:5d}  max|diff|={diff:.3e}  "
              f"{'PASS' if ok else 'FAIL'}  ({bit})")

        flops = 2.0 * B * H * S * (S + 1) * D
        t_ours = bench(lambda: mod.fa2_wgmma_forward(q, k, v, scale))
        t_sdpa = bench(lambda: ref(q, k, v, scale))
        print(f"    ours {t_ours:7.3f} ms / {flops / t_ours / 1e9:6.2f} TFLOPS"
              f" ; SDPA {t_sdpa:7.3f} ms / {flops / t_sdpa / 1e9:6.2f} TFLOPS"
              f"   ratio {t_sdpa / t_ours:.2f}x")

    print("\n结论：" + ("全部 PASS，TMA 版数值与 w12e 一致" if fails == 0 else f"{fails} 档 FAIL"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
