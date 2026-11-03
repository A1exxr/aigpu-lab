"""w11 FlashAttention 反向 · 台阶 1：单块版（S=64）

一个 program 负责一个 (b,h)，不循环、不 atomic —— 只验证「公式 → Triton」翻译没错。
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


if __name__ == "__main__":
    test_step1()
