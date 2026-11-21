# -*- coding: utf-8 -*-
"""w12g（FA2 前向，warp specialization 版）：正确性三档 + 与 SDPA 比性能。

结构 = 1 个 producer warpgroup（只发 TMA）+ 2 个 consumer warpgroup（只算），
384 线程，用 mbarrier(full/empty) 做生产者-消费者握手。

host 侧函数名与 w12e/w12f 相同：`fa2_wgmma_forward`（在 .cu 里没改名，避免动用户代码）。

需要 compute_cap >= 9.0 的卡（H100/H800/H200/H20）。本地 4060 编译就会报
"arch=sm_90a not supported"，那是预期的——只在 Hopper 上跑本脚本。

跑法（租机）：
    python kernels/wgmma_warmup.py        # 1) 描述符语义（3 分钟，错了后面全错）
    python kernels/w12e_fa2_wgmma.py      # 2) QK=wgmma，K/V 走 cp.async
    python kernels/w12f_fa2_tma.py        # 3) K/V 也走 TMA
    python kernels/w12g_fa2_ws.py         # 4) 本文件：再叠 warp specialization
三档 max|diff| 必须与 w12e 位级一致：3.644e-04 / 4.944e-04 / 5.429e-04。
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

CU = Path(__file__).with_name("w12g_fa2_ws.cu").read_text(encoding="utf-8")

CPP = "torch::Tensor fa2_wgmma_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, double scale);"
FUNC = "fa2_wgmma_forward"

# 三档与 w12e/w12f 完全相同，便于逐档对比 diff 是否位级一致
CASES = [(2, 8, 2, 512, 64), (2, 8, 2, 125, 64), (2, 8, 2, 1024, 64)]
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
        name="w12g_fa2_ws",
        cpp_sources=CPP,
        cuda_sources=CU,
        functions=[FUNC],
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
    fn = getattr(mod, FUNC)
    torch.manual_seed(0)
    fails = 0
    for i, (B, H, Hkv, S, D) in enumerate(CASES):
        q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, Hkv, S, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, Hkv, S, D, device="cuda", dtype=torch.float16)
        scale = 1.0 / (D ** 0.5)

        ours = fn(q, k, v, scale)
        r = ref(q, k, v, scale)
        diff = (ours.float() - r.float()).abs().max().item()
        bit = "位级一致" if abs(diff - EXPECT[i]) < 1e-9 else f"与 w12e 的 {EXPECT[i]:.3e} 不同"
        ok = diff <= 3e-3
        fails += 0 if ok else 1
        print(f"B={B} H={H} Hkv={Hkv} S={S:5d}  max|diff|={diff:.3e}  "
              f"{'PASS' if ok else 'FAIL'}  ({bit})")

        flops = 2.0 * B * H * S * (S + 1) * D
        t_ours = bench(lambda: fn(q, k, v, scale))
        t_sdpa = bench(lambda: ref(q, k, v, scale))
        print(f"    ours {t_ours:7.3f} ms / {flops / t_ours / 1e9:6.2f} TFLOPS"
              f" ; SDPA {t_sdpa:7.3f} ms / {flops / t_sdpa / 1e9:6.2f} TFLOPS"
              f"   ratio {t_sdpa / t_ours:.2f}x")

    print("\n结论：" + ("全部 PASS，warp specialization 版数值与 w12e 一致"
                       if fails == 0 else f"{fails} 档 FAIL"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
