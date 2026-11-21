# -*- coding: utf-8 -*-
"""w12e（FA2 前向，QK 用 wgmma）：正确性三档 + 与 SDPA 比性能。
需要 sm_90a 卡（H100/H800/H20）。本地 4060 会在编译处报错，这是预期的。"""
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

CU = Path(__file__).with_name("w12e_fa2_wgmma.cu").read_text(encoding="utf-8")
CPP = "torch::Tensor fa2_wgmma_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, double scale);"

mod = load_inline(
    name="w12e_fa2_wgmma",
    cpp_sources=CPP,
    cuda_sources=CU,
    functions=["fa2_wgmma_forward"],
    extra_cuda_cflags=["-arch=sm_90a", "-O3", "--use_fast_math"],
    verbose=False,
)

CASES = [(2, 8, 2, 512, 64), (2, 8, 2, 125, 64), (2, 8, 2, 1024, 64)]


def ref(q, k, v, scale):
    kk = k.repeat_interleave(q.size(1) // k.size(1), dim=1)
    vv = v.repeat_interleave(q.size(1) // k.size(1), dim=1)
    return F.scaled_dot_product_attention(q, kk, vv, is_causal=True, scale=scale)


def bench(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    beg = torch.cuda.Event(True)
    end = torch.cuda.Event(True)
    beg.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return beg.elapsed_time(end) / iters


def main():
    torch.manual_seed(0)
    for B, H, Hkv, S, D in CASES:
        q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, Hkv, S, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, Hkv, S, D, device="cuda", dtype=torch.float16)
        scale = 1.0 / (D ** 0.5)

        ours = mod.fa2_wgmma_forward(q, k, v, scale)
        r = ref(q, k, v, scale)
        diff = (ours.float() - r.float()).abs().max().item()
        ok = "PASS" if diff <= 3e-3 else "FAIL"
        print(f"B={B} H={H} Hkv={Hkv} S={S:5d}  max|diff|={diff:.3e}  {ok}")

        flops = 2.0 * B * H * S * (S + 1) * D
        t_ours = bench(lambda: mod.fa2_wgmma_forward(q, k, v, scale))
        t_sdpa = bench(lambda: ref(q, k, v, scale))
        print(f"    ours {t_ours:7.3f} ms / {flops / t_ours / 1e9:6.2f} TFLOPS"
              f" ; SDPA {t_sdpa:7.3f} ms / {flops / t_sdpa / 1e9:6.2f} TFLOPS"
              f"   ratio {t_sdpa / t_ours:.2f}x")


if __name__ == "__main__":
    main()
