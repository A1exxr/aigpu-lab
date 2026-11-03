"""w11 FlashAttention 反向 · 台阶 3：拆两 kernel，无 atomic

dk/dv 与 dq 各自独占一维 → 循环里累加进寄存器，结束一次 store；结果逐位可复现。
"""

import math

import torch
import triton
import triton.language as tl

from w11_oracle import bwd_ref, fwd, make_inputs


# ============================================================
# 台阶 3：key 维也并行 —— 拆成两个 kernel，各自独占一维
#   3a _bwd_dkdv : grid=(S//BLOCK_N, B*H)  串行走 query 块 → dk, dv 独占 store
#   3b _bwd_dq   : grid=(S//BLOCK_M, B*H)  串行走 key   块 → dq    独占 store
# 没有 atomic。代价：p / ds 要算两遍（约 +50% FLOPs）。
# ============================================================

@triton.jit
def _bwd_dkdv(Q, K, V, DO, LSE, DELTA, DK, DV,
              H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
              scale, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # B, H, S, D = 1, 8, 1024, 64
    pid_n = tl.program_id(0)                    # 0 1 2 ... 15
    bh = tl.program_id(1)                       # 0 1 ... 7   ———— B * H = 8  
    b = bh // H                                 # 0
    h = bh % H                                  # 0 1 2 ... 7

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # 当前 program 负责的 key 行
    offs_d = tl.arange(0, D)
    base = (b * H + h) * S * D

    k = tl.load(K + base + offs_n[:, None] * D + offs_d[None, :])   # (BN, D)
    v = tl.load(V + base + offs_n[:, None] * D + offs_d[None, :])   # (BN, D)

    # 两个累加器 —— 形状要和 dk/dv 的“本块”一致
    acc_dk = tl.zeros((BLOCK_N, D), dtype = tl.float32)        
    acc_dv = tl.zeros((BLOCK_N, D), dtype = tl.float32)

    # 循环步长是 query 块（这一维现在是串行的）。range 的三个参数填全
    for m in range(0, S, BLOCK_M):
        offs_m = m + tl.arange(0, BLOCK_M)
        q = tl.load(Q + base + offs_m[:, None] * D + offs_d[None, :])     # (BM, D)
        do = tl.load(DO + base + offs_m[:, None] * D + offs_d[None, :])   # (BM, D)
        lse_m = tl.load(LSE + bh * S + offs_m)        # (BM,)
        delta_m = tl.load(DELTA + bh * S + offs_m)    # (BM,)

        s = tl.dot(q, tl.trans(k), input_precision="ieee") * scale            # (BM, BN)
        p = tl.exp(s - lse_m[:, None])                # (BM, BN)
        dp = tl.dot(do, tl.trans(v), input_precision="ieee")                  # (BM, BN)
        ds = p * (dp - delta_m[:, None])              # (BM, BN)

        # dk 累加
        acc_dk += tl.dot(tl.trans(ds), q, input_precision="ieee") * scale
        # dv 累加
        acc_dv += tl.dot(tl.trans(p), do, input_precision="ieee")

    # 把 acc_dk / acc_dv 写回 DK / DV（指针形状和 dk/dv 的块一致）
    # 凡是"行号/token 编号/offs_m/offs_n/offs_k"这类走行方向的索引，都要 [:, None] 竖起来；
    # 凡是"列号/特征编号/offs_d"这类走列方向的索引，都要 [None, :] 横过来
    tl.store(DK + base + offs_n[:,None] * D + offs_d[None,:], acc_dk)
    tl.store(DV + base + offs_n[:,None] * D + offs_d[None,:], acc_dv) 


@triton.jit
def _bwd_dq(Q, K, V, DO, LSE, DELTA, DQ,
            H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
            scale, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # 当前 program 负责的 query 行
    offs_d = tl.arange(0, D)
    base = (b * H + h) * S * D

    q = tl.load(Q + base + offs_m[:, None] * D + offs_d[None, :])     # (BM, D)
    do = tl.load(DO + base + offs_m[:, None] * D + offs_d[None, :])   # (BM, D)
    lse_m = tl.load(LSE + bh * S + offs_m)        # (BM,)
    delta_m = tl.load(DELTA + bh * S + offs_m)    # (BM,)

    # dq 的累加器
    acc_dq = tl.zeros((BLOCK_M, D), dtype = tl.float32)

    for n in range(0, S, BLOCK_N):
        offs_n = n + tl.arange(0, BLOCK_N)
        k = tl.load(K + base + offs_n[:, None] * D + offs_d[None, :])   # (BN, D)
        v = tl.load(V + base + offs_n[:, None] * D + offs_d[None, :])   # (BN, D)

        s = tl.dot(q, tl.trans(k), input_precision="ieee") * scale           # (BM, BN)
        p = tl.exp(s - lse_m[:, None])               # (BM, BN)
        dp = tl.dot(do, tl.trans(v), input_precision="ieee")                 # (BM, BN)
        ds = p * (dp - delta_m[:, None])             # (BM, BN)

        # dq 累加
        acc_dq += tl.dot(ds, k, input_precision="ieee") * scale

    # 写回 DQ
    tl.store(DQ + base + offs_m[:,None] * D + offs_d[None,:], acc_dq)


def attn_bwd_triton_split(q, k, v, do, lse, delta, BLOCK_M=64, BLOCK_N=64):
    B, H, S, D = q.shape
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    scale = 1.0 / math.sqrt(D)

    # 注意两个 grid 不一样：3a 只切 key，3b 只切 query
    # num_stages=2：默认 3 级时 k+v 各 16 KB × 3 + 中间缓冲 = 114688 B > 上限 101376 B
    # （E-97 那道账）。降到 2 级后 64 KB + 中间缓冲，刚好装下。
    _bwd_dkdv[(S // BLOCK_N, B * H)](
        q, k, v, do, lse, delta, dk, dv,
        H=H, S=S, D=D, scale=scale, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_stages=2,
    )
    _bwd_dq[(S // BLOCK_M, B * H)](
        q, k, v, do, lse, delta, dq,
        H=H, S=S, D=D, scale=scale, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_stages=2,
    )
    return dq, dk, dv


def test_step3():
    print()
    print("=" * 70)
    print("台阶 3：key 维并行（两 kernel，无 atomic）S=1024 fp32 判据 1e-5")
    print("=" * 70)
    B, H, S, D = 1, 8, 1024, 64
    q, k, v, do = make_inputs(B, H, S, D, seed=3, dtype=torch.float32)

    s_f, p_f, o, lse_f = fwd(q, k, v)
    dq_ref, dk_ref, dv_ref, p_ref, lse_ref, ds_ref = bwd_ref(q, k, v, do)
    delta = (do * o).sum(dim=-1)      # 分母已经验证过，这里直接用

    dq, dk, dv = attn_bwd_triton_split(q, k, v, do, lse_f, delta)

    ok = True
    for name, mine, ref in (("dq", dq, dq_ref), ("dk", dk, dk_ref), ("dv", dv, dv_ref)):
        err = (mine - ref).abs().max().item()
        good = err < 1e-5
        ok &= good
        print(f"{name}: max_abs = {err:.3e}   {'OK' if good else 'FAIL'}")

    # 可复现性：连跑两次必须逐位相同（没有 atomic 才有这个性质）
    dq2, dk2, dv2 = attn_bwd_triton_split(q, k, v, do, lse_f, delta)
    same = torch.equal(dq, dq2) and torch.equal(dk, dk2) and torch.equal(dv, dv2)
    print(f"两次结果完全一致: {'OK' if same else 'FAIL'}")
    ok &= same

    print("\n" + ("台阶 3 通过：无 atomic、可复现 ✅" if ok else "台阶 3 未通过 —— 先看 dk/dv（累加器）再看 dq"))
    return ok


if __name__ == "__main__":
    test_step3()
