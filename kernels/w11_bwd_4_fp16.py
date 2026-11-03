"""w11 FlashAttention 反向 · 台阶 4：fp16 + Tensor Core

输入 fp16、dot 出 fp32、判据放宽到 2e-3；lse/delta 保持 fp32（O(S) 小数组）。
"""

import math

import torch
import triton
import triton.language as tl

from w11_oracle import bwd_ref, fwd, make_inputs
from w11_bwd_3_split import attn_bwd_triton_split   # bench_step4 里要对 fp32 版计时


# ============================================================
# 台阶 4：fp16 + Tensor Core（判据放宽到 2e-3）
#   换 fp16 只改三处：输入 dtype、dot 的 out_dtype、判据
#   注意：s 一定要以 fp32 出来，否则 exp 之前就丢精度
# ============================================================
#  输入变 fp16，但 lse、delta 保持 fp32 它们是 O(S) 的小数组
@triton.jit
def _bwd_dkdv_fp16(Q, K, V, DO, LSE, DELTA, DK, DV,
                   H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
                   scale, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_n = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    base = (b * H + h) * S * D

    k = tl.load(K + base + offs_n[:, None] * D + offs_d[None, :])           # fp16
    v = tl.load(V + base + offs_n[:, None] * D + offs_d[None, :])           # fp16

    acc_dk = tl.zeros((BLOCK_N, D), dtype=tl.float32)
    acc_dv = tl.zeros((BLOCK_N, D), dtype=tl.float32)

    for m in range(0, S, BLOCK_M):
        offs_m = m + tl.arange(0, BLOCK_M)
        q = tl.load(Q + base + offs_m[:, None] * D + offs_d[None, :])       # fp16
        do = tl.load(DO + base + offs_m[:, None] * D + offs_d[None, :])     # fp16
        lse_m = tl.load(LSE + bh * S + offs_m)                              # fp32（小数组，不动）
        delta_m = tl.load(DELTA + bh * S + offs_m)                          # fp32

        # q/k/do/v 是 fp16，tl.dot 走 fp16 Tensor Core；out_dtype=tl.float32 让累加器是 fp32
        # 这是 fp16 GEMM 的标准正确姿势：输入 fp16（省带宽、上 TC），累加 fp32（保精度）
        # 而且s必须要走fp32，fp16 在 0~30 的量级上只有 ~1e-2 的绝对误差，进 exp 之前就炸了
        s = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale            # (BM, BN) fp32
        p = tl.exp(s - lse_m[:, None])                                      # fp32
        dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32)                  # fp32
        ds = p * (dp - delta_m[:, None])                                    # fp32

        # 这两个 dot 的输入是 fp32（ds/p）—— Triton 会退回 CUDA Core；
        # 想全走 Tensor Core 就得接受 ds 也降 fp16，但精度会掉到 ~1e-2，
        acc_dk += tl.dot(tl.trans(ds), q.to(tl.float32)) * scale
        acc_dv += tl.dot(tl.trans(p), do.to(tl.float32))

    tl.store(DK + base + offs_n[:, None] * D + offs_d[None, :], acc_dk.to(DK.dtype.element_ty))
    tl.store(DV + base + offs_n[:, None] * D + offs_d[None, :], acc_dv.to(DV.dtype.element_ty))


@triton.jit
def _bwd_dq_fp16(Q, K, V, DO, LSE, DELTA, DQ,
                 H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
                 scale, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    base = (b * H + h) * S * D

    q = tl.load(Q + base + offs_m[:, None] * D + offs_d[None, :])
    do = tl.load(DO + base + offs_m[:, None] * D + offs_d[None, :])
    lse_m = tl.load(LSE + bh * S + offs_m)
    delta_m = tl.load(DELTA + bh * S + offs_m)

    acc_dq = tl.zeros((BLOCK_M, D), dtype=tl.float32)

    for n in range(0, S, BLOCK_N):
        offs_n = n + tl.arange(0, BLOCK_N)
        k = tl.load(K + base + offs_n[:, None] * D + offs_d[None, :])
        v = tl.load(V + base + offs_n[:, None] * D + offs_d[None, :])

        s = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale
        p = tl.exp(s - lse_m[:, None])
        dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32)
        ds = p * (dp - delta_m[:, None])

        acc_dq += tl.dot(ds, k.to(tl.float32)) * scale

    tl.store(DQ + base + offs_m[:, None] * D + offs_d[None, :], acc_dq.to(DQ.dtype.element_ty))


def attn_bwd_triton_split_fp16(q, k, v, do, lse, delta, BLOCK_M=64, BLOCK_N=64):
    B, H, S, D = q.shape
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    scale = 1.0 / math.sqrt(D)

    _bwd_dkdv_fp16[(S // BLOCK_N, B * H)](
        q, k, v, do, lse, delta, dk, dv,
        H=H, S=S, D=D, scale=scale, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_stages=2,
    )
    _bwd_dq_fp16[(S // BLOCK_M, B * H)](
        q, k, v, do, lse, delta, dq,
        H=H, S=S, D=D, scale=scale, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_stages=2,
    )
    return dq, dk, dv


def _step4_data(B=1, H=8, S=1024, D=64, seed=4):
    """同一组数据造两份：fp32 当参考答案，fp16 喂 kernel。"""
    q32, k32, v32, do32 = make_inputs(B, H, S, D, seed=seed, dtype=torch.float32)
    s_f, p_f, o32, lse32 = fwd(q32, k32, v32)
    dq_ref, dk_ref, dv_ref, *_ = bwd_ref(q32, k32, v32, do32)
    delta32 = (do32 * o32).sum(dim=-1)
    q, k, v, do = (t.half() for t in (q32, k32, v32, do32))
    return (q, k, v, do, lse32, delta32), (dq_ref, dk_ref, dv_ref), (q32, k32, v32, do32, o32, lse32)


def test_step4():
    print()
    print("=" * 70)
    print("台阶 4：fp16 + Tensor Core   判据 max_abs < 2e-3")
    print("=" * 70)
    (q, k, v, do, lse, delta), (dq_ref, dk_ref, dv_ref), _ = _step4_data()

    dq, dk, dv = attn_bwd_triton_split_fp16(q, k, v, do, lse, delta)

    ok = True
    for name, mine, ref in (("dq", dq, dq_ref), ("dk", dk, dk_ref), ("dv", dv, dv_ref)):
        err = (mine.float() - ref).abs().max().item()
        good = err < 2e-3
        ok &= good
        print(f"{name}: max_abs = {err:.3e}   {'OK' if good else 'FAIL'}   (判据 2e-3)")

    print("\n" + ("台阶 4 通过：fp16 通道可用 ✅" if ok else "台阶 4 未通过 —— 先看是不是哪一步漏了 out_dtype=fp32"))
    return ok


def bench_step4():
    (q, k, v, do, lse, delta), _, (q32, k32, v32, do32, o32, lse32) = _step4_data()
    delta32 = delta

    def bench(fn, rounds=3):
        return min(triton.testing.do_bench(fn, warmup=25, rep=100) for _ in range(rounds))

    t3 = bench(lambda: attn_bwd_triton_split(q32, k32, v32, do32, lse32, delta32))
    t4 = bench(lambda: attn_bwd_triton_split_fp16(q, k, v, do, lse, delta))

    B, H, S, D = q.shape
    flops = 6.0 * B * H * S * S * D          # 台阶3/4 的口径：p、ds 各算两遍

    print()
    print("=" * 64)
    print(f"{'实现':<26}{'ms':>10}{'TFLOPS':>12}{'相对台阶3':>14}")
    print("-" * 64)
    print(f"{'台阶3（fp32+ieee）':<26}{t3:>10.3f}{flops/t3/1e9:>12.2f}{1.0:>14.2f}")
    print(f"{'台阶4（fp16）':<26}{t4:>10.3f}{flops/t4/1e9:>12.2f}{t4/t3:>14.2f}")
    print("=" * 64)
    print(f"fp16 提速 = {t3/t4:.2f}×    理论峰值（fp16 TC，本卡）≈ 65 TFLOPS")
    print(f"台阶4 时间 = {t4:.3f} ms   （台阶4 的目标线 ≤ 1.5 ms）")


if __name__ == "__main__":
    test_step4()
    bench_step4()
