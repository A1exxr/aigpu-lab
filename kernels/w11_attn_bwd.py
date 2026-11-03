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
    # tl.dot收到fp32输入时，默认不走真fp32，而是把位数从23位砍到10位(tf32)，这样就过不了1e-5 的误差判断
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
    test_step1()
    test_step2()
    test_step3()
    test_step4()
    bench_step4()