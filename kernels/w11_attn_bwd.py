"""w11 FlashAttention 反向 · 台阶 1：Triton 单块版

台阶 0（PyTorch 参考实现）已经搬到 `w11_oracle.py`，这里只放 Triton kernel。

台阶 1：S=64，一个 program 负责一个 (b,h)，不做分块循环、不用 atomic。
          目的只有一个 —— 验证"公式 → Triton 代码"翻译没错。


命名统一（跟 CUDA 代码和 w11_oracle.py 完全一致，别再换）：
    B, H, S, D   —— 维度：batch / head 数 / 序列长 / 每头维度
    q, k, v, do, lse, s, p, dp, ds, dq, dk, dv —— 与台阶 0 同名
    将来分块时才引入 tile 名：BLOCK_M, BLOCK_N（块大小），不叫 SEQ/DH


"""

import math
import torch
import triton
import triton.language as tl

from w11_oracle import bwd_ref, fwd, make_inputs


@triton.jit
def _attn_bwd_kernel(Q, K, V, DO, LSE, DQ, DK, DV,
                     H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
                     scale: tl.constexpr):
    pid = tl.program_id(0)              # 0 1 2 ... 7
    b = pid // H                        # B = 1，H = 8
    h = pid % H
    base = (b * H + h) * S * D                  # 这个 (b,h) 的矩阵起点

    m = tl.arange(0, S)                         # 行索引 0 1 2 ... 63
    d = tl.arange(0, D)                         # 列索引 0 1 2 ... 63
    ptr = base + m[:, None] * D + d[None, :]    # (S,D) 的偏移模板

    # ---- 载入 ----
    q  = tl.load(Q  + ptr)                       # (S,D)
    k  = tl.load(K  + ptr)                       # (S,D)
    v  = tl.load(V  + ptr)                       # (S,D)
    do = tl.load(DO + ptr)                       # (S,D)
    # lse形状是(B, H, S)
    lse = tl.load(LSE + (b * H + h) * S + m)     # (S,)


    # 这里是triton核函数，不能用torch的方法
    # 这里的s，是softmax之前的s，原始分数，前向没有保存下来，太大了
    # 前向只保存了 lse 和 o
    # tl.dot就是矩阵乘法，tl.trans()默认交换最后两维，传参可以支持任意维度交换
    # ieee 强制使用 完整的 IEEE 754 单精度（FP32） 进行乘加运算 ———— 绕过tensor core 使用 cuda core
    s = tl.dot(q, tl.trans(k), input_precision="ieee") * scale

    p = tl.exp(s - lse[:,None])      

    # dv = p.transpose(-1, -2) @ do    不能用torch的方法
    dv = tl.dot(tl.trans(p), do, input_precision="ieee")

    dp = tl.dot(do, tl.trans(v), input_precision="ieee")

    # delta = torch.sum(dp * p, dim=-1, keepdim=True)
    delta = tl.sum(dp * p, axis = 1)    # 没有keepdim方法，下面就要补齐维度

    ds = p * (dp - delta[:,None])

    # dq = ds @ k / math.sqrt(scale)
    dq = tl.dot(ds, k, input_precision="ieee") * scale

    # dk = ds.transpose(-1, -2) * q / math.sqrt(scale)
    dk = tl.dot(tl.trans(ds), q, input_precision="ieee") * scale

    tl.store(DQ + ptr, dq)
    tl.store(DK + ptr, dk)
    tl.store(DV + ptr, dv)


def attn_bwd_triton(q, k, v, do, lse):
    """启动封装"""
    B, H, S, D = q.shape                                        #  B, H, S, D = 1, 8, 64, 64
    q, k, v, do = (x.contiguous() for x in (q, k, v, do))
    lse = lse.reshape(B, H, S).contiguous()
    dq, dk, dv = (torch.empty_like(q) for _ in range(3))
    _attn_bwd_kernel[(B * H,)](                                 # grid大小为(8,)，GPU 会启动 8 个并发实例。
        q, k, v, do, lse, dq, dk, dv,
        H, S, D, 1.0 / math.sqrt(D),
    )
    return dq, dk, dv


def test_step1():
    print("=" * 70)
    print("台阶 1：Triton 单块版（S=64，fp32，判据 max_abs < 1e-5）")
    print("=" * 70)
    B, H, S, D = 1, 8, 64, 64
    q, k, v, do = make_inputs(B, H, S, D, seed=1, dtype=torch.float32)

    dq_ref, dk_ref, dv_ref, p_ref, lse, ds_ref = bwd_ref(q, k, v, do)
    dq, dk, dv = attn_bwd_triton(q, k, v, do, lse)

    print(f"配置 B={B} H={H} S={S} D={D}  dtype=fp32")
    ok = True
    for name, mine, ref in (("dq", dq, dq_ref), ("dk", dk, dk_ref), ("dv", dv, dv_ref)):
        err = (mine - ref).abs().max().item()
        good = err < 1e-5
        ok &= good
        print(f"{name}: max_abs = {err:.3e}   {'OK' if good else 'FAIL'}")

    print("\n" + ("台阶 1 通过：公式翻译没错 ✅" if ok else "台阶 1 未通过 —— 先把最错的那个单独查"))
    return ok


# ==========================================================================
#   台阶 2：分块版（S=1024，BLOCK_M=64 行一块，循环 BLOCK_N 列，用 atomic 累加 dk/dv）
#
#  和台阶 1 的三个区别：
#    1. 一个 program 只负责 BLOCK_M 行 → 只装 (BLOCK_M,BLOCK_N) 的分数块，不装整行 → 必须循环 j
#    2. dq 只属于这 BLOCK_M 行 → 直接 store；dk/dv 会被**所有** pid_m 碰到 → atomic_add
#    3. delta 不能再由 dp∘p 算（S² 存不下）→ 先跑 preprocess kernel 算成 (S,) 小数组
# ==========================================================================

# 16个program各切 64 行、能跑大 S，但代价是三件事——循环、atomic、preprocess
# 台阶 2 只切了行（query 行），key 维没有并行，只是在循环里串行走过
# 在主kernel之前，用一个小kernel算好delta  预处理
# delta_i 的求和跨所有 16 个 j 块，而主kernel里的循环一次只给你 1 个 j 块，算不出来，所以必须在循环外先算好
@triton.jit
def _bwd_preprocess(O, DO, DELTA,
                    H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
                    BLOCK_M: tl.constexpr):
    # grid 是 (16, 8)    128个program，每个 program 负责1个头的64行
    pid_m = tl.program_id(0)                # 0 1 2 ... 15  行块编号
    pid_bh = tl.program_id(1)               # 0 1 ... 7    负责哪个(b,h)块
    b = pid_bh // H                         # B=1，b就是0
    h = pid_bh % H                          # 头的编号
    base = (b * H + h) * S * D              # 这个 (b,h) 头的起点
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)                #  全局行索引，本 program 负责的行段起点
    offs_d = tl.arange(0, D)                                        # D 维列索引，D 维没有分块
    # offs_m[:, None] * D ———— 每行的起始偏移（行号 × D）
    # offs_d[None, :]   ———— 行内列偏移
    ptr = base + offs_m[:, None] * D + offs_d[None, :]  

    o  = tl.load(O  + ptr)                       # (BLOCK_M,D)  (64, 64)
    do = tl.load(DO + ptr)                       # (BLOCK_M,D)  (64, 64)

    # delta 形状 (BLOCK_M,) —— 逐元素乘完沿 D 求和
    delta = tl.sum(o * do, axis=1)

    tl.store(DELTA + (b * H + h) * S + offs_m, delta)       # 完整 delta 形状 (B, H, S)  (1, 8, 1024)


@triton.jit
def _attn_bwd_chunked(Q, K, V, DO, LSE, DELTA, DQ, DK, DV,
                      H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, scale: tl.constexpr):
    pid_m = tl.program_id(0)                     # 第几个行块：0 1 2 ... 15
    pid_bh = tl.program_id(1)                    # 哪个 (b,h)
    b = pid_bh // H
    h = pid_bh % H
    base = (b * H + h) * S * D

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)        
    offs_n = tl.arange(0, BLOCK_N)                          
    offs_d = tl.arange(0, D)                    # 特征维度 0 1 2 ... 63

    q  = tl.load(Q  + base + offs_m[:, None] * D + offs_d[None, :])    # (BLOCK_M,D)
    do = tl.load(DO + base + offs_m[:, None] * D + offs_d[None, :])    # (BLOCK_M,D)
    lse   = tl.load(LSE   + (b * H + h) * S + offs_m)                  # (BLOCK_M,)     (B, H, S)
    delta = tl.load(DELTA + (b * H + h) * S + offs_m)                  # (BLOCK_M,)     (B, H, S)

    # 要固定Q块i，遍历所有K/V块j，才能算出dq
    acc_dq = tl.zeros((BLOCK_M, D), dtype=tl.float32)                       # dq 的累加器

    # 这里还是串行，key维度没有并行
    for j in range(0, S // BLOCK_N):
        cols = j * BLOCK_N + offs_n                                        # 本块的列号 (BLOCK_N,)
        k = tl.load(K + base + cols[:, None] * D + offs_d[None, :])   # (BLOCK_N,D)
        v = tl.load(V + base + cols[:, None] * D + offs_d[None, :])   # (BLOCK_N,D)

        # ========================

        s = tl.dot(q, tl.trans(k), input_precision="ieee") * scale      # (BLOCK_M, BLOCK_N)

        p = tl.exp(s - lse[:,None])                                     # (BLOCK_M, BLOCK_N)

        dv_j = tl.dot(tl.trans(p), do, input_precision="ieee")          # (BLOCK_N, D)

        dp = tl.dot(do, tl.trans(v), input_precision="ieee")            # (BLOCK_M, BLOCK_N)

        ds = p * (dp - delta[:,None])                                   # (BLOCK_M, BLOCK_N)

        # acc_dq 累加这一块的贡献
        acc_dq += tl.dot(ds, k, input_precision="ieee") * scale         # (BLOCK_N, D)

        # dk_j 形状 (BLOCK_N,D) —— 这一块对 dk 的贡献
        dk_j = tl.dot(tl.trans(ds), q, input_precision="ieee") * scale  # (BLCO_N, D)

        # ============  ============

        # dk/dv 的这 BLOCK_N 行会被所有 pid_m 碰到 → 必须 atomic_add
        tl.atomic_add(DK + base + cols[:, None] * D + offs_d[None, :], dk_j)
        tl.atomic_add(DV + base + cols[:, None] * D + offs_d[None, :], dv_j)

    # dq 只属于这 BLOCK_M 行 → 直接写
    tl.store(DQ + base + offs_m[:, None] * D + offs_d[None, :], acc_dq)


def attn_bwd_triton_chunked(q, k, v, o, do, lse, BLOCK_M=64, BLOCK_N=64):
    """ 分块版启动封装

    注意：dk/dv 用 atomic_add 累加，所以必须先 torch.zeros 清零，否则会加到垃圾上。
    """
    B, H, S, D = q.shape                                            # B, H, S, D = 1, 8, 1024, 64
    q, k, v, o, do = (x.contiguous() for x in (q, k, v, o, do))
    lse = lse.reshape(B, H, S).contiguous()
    scale = 1.0 / math.sqrt(D)

    delta = torch.empty((B, H, S), device=q.device, dtype=torch.float32)
    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)

    # grid 是 (16, B*H)
    _bwd_preprocess[(S // BLOCK_M, B * H)](o, do, delta, H, S, D, BLOCK_M)

    _attn_bwd_chunked[(S // BLOCK_M, B * H)](
        q, k, v, do, lse, delta, dq, dk, dv,
        H, S, D, BLOCK_M, BLOCK_N, scale,
        num_stages=2,      # 默认 3 会在 fp32 下爆共享内存（每级缓冲一份 k/v 块）
    )
    return dq, dk, dv, delta


def test_step2():
    print()
    print("=" * 70)
    print("台阶 2：分块版（S=1024，BLOCK_M=BLOCK_N=64，fp32，判据 max_abs < 1e-5）")
    print("=" * 70)
    B, H, S, D = 1, 8, 1024, 64
    q, k, v, do = make_inputs(B, H, S, D, seed=2, dtype=torch.float32)

    s_f, p_f, o, lse_f = fwd(q, k, v)
    dq_ref, dk_ref, dv_ref, p_ref, lse_ref, ds_ref = bwd_ref(q, k, v, do)
    delta_alt = (do * o).sum(dim=-1)          # 独立算一遍，验证 preprocess kernel

    dq, dk, dv, delta = attn_bwd_triton_chunked(q, k, v, o, do, lse_f)

    print(f"配置 B={B} H={H} S={S} D={D}  dtype=fp32  grid=({S // 64}, {B * H})")
    de = (delta - delta_alt).abs().max().item()
    print(f"delta（preprocess kernel）: max_abs = {de:.3e}   {'OK' if de < 1e-5 else 'FAIL'}")

    ok = de < 1e-5
    for name, mine, ref in (("dq", dq, dq_ref), ("dk", dk, dk_ref), ("dv", dv, dv_ref)):
        err = (mine - ref).abs().max().item()
        good = err < 1e-5
        ok &= good
        print(f"{name}: max_abs = {err:.3e}   {'OK' if good else 'FAIL'}")

    print("\n" + ("台阶 2 通过：分块 + atomic 正确 ✅" if ok else "台阶 2 未通过 —— delta 先看，再看 dk/dv（atomic 最容易错）"))
    return ok


if __name__ == "__main__":
    test_step1()
    test_step2()