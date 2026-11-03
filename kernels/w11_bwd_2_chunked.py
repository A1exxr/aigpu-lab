"""w11 FlashAttention 反向 · 台阶 2：分块 + atomic（S=1024）

每个 program 负责 BLOCK_M 行，循环走 key 块；dq 直接 store，dk/dv 用 atomic_add 累加。
"""

import math

import torch
import triton
import triton.language as tl

from w11_oracle import bwd_ref, fwd, make_inputs


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

    # 循环外：载入一次，整个 kernel 不变
    q  = tl.load(Q  + base + offs_m[:, None] * D + offs_d[None, :])    # (BLOCK_M,D)
    do = tl.load(DO + base + offs_m[:, None] * D + offs_d[None, :])    # (BLOCK_M,D)
    lse   = tl.load(LSE   + (b * H + h) * S + offs_m)                  # (BLOCK_M,)     (B, H, S)
    delta = tl.load(DELTA + (b * H + h) * S + offs_m)                  # (BLOCK_M,)     (B, H, S)

    # 要固定Q块i，遍历所有K/V块j，才能算出dq
    acc_dq = tl.zeros((BLOCK_M, D), dtype=tl.float32)                       # dq 的累加器，在寄存器里，没有竞争，不需要用atomic_add

    # 这里还是串行，key维度没有并行
    # 循环内，每一轮都会换一批k/v
    for j in range(0, S // BLOCK_N):        # j = 0 1 2 ... 15，每个program都会遍历完S
        # 这 16 个program是并行的，并且都遍历全部的key，地址重叠，且都有写操作，会竞争(如果只有读操作，就不会竞争)
        # 所以 dk/dv 的每个 key 块会被全部 16 个 pid_m 的 program 在各自循环里碰到，有竞争，必须 atomic_add
        # j=0 → cols = [0..63]；j=1 → [64..127]；...；j=15 → [960..1023]  本轮64个key token的全局编号
        # 这64个数也是key token在物理[S,D]里的行号，1024行
        cols = j * BLOCK_N + offs_n                                     # 本块的列号 (BLOCK_N,)，这里的列号，是在分数矩阵里面的名字(实际上就是key维度的行号)
        k = tl.load(K + base + cols[:, None] * D + offs_d[None, :])     # (BLOCK_N,D)
        v = tl.load(V + base + cols[:, None] * D + offs_d[None, :])     # (BLOCK_N,D)

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
        # Triton 的 tl.load 把数据搬进 shared memory 暂存，再喂给 tl.dot
        # 开启软件流水线：为隐藏内存延迟，Triton 把循环里的 load 预取 3 份（第 0 份在算，后 2 份在等/传）
        num_stages=2,      
        # 流水线若是 3 会在 fp32 下爆共享内存（每级缓冲一份 k/v 块）
        # 单个k_j/v_j块为64 * 64 * 4B = 16KB，那k流水线为3*16=48KB，v流水线也是48kb ，合计为96kb，而共享内存本来就100kb大小
        # q_i在循环中是不变的量 ———— 循环不变量不参与流水线，所以只占 1 份 16 KB
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
    test_step2()
