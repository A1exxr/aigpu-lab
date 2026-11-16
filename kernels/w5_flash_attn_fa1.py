# -*- coding: utf-8 -*-
"""
    FlashAttention V1：
        FA1 的 grid 只有(B * H)，并行度只有这么大，一个 program 负责一对 (b,h) 的整个序列
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _flash_attn_kernel(Q, K, V, O,
                       stride_qh, stride_qm, stride_qd,
                       stride_kh, stride_km, stride_kd,
                       stride_vh, stride_vm, stride_vd,
                       stride_oh, stride_om, stride_od,
                       S, scale,
                       Block_m: tl.constexpr, Block_n: tl.constexpr, D: tl.constexpr):
    pid_ph = tl.program_id(0)             # FA1：grid 只有 B*H 一维，没有 pid_m
    offs_d = tl.arange(0, D)              # (D,)

    # ===== 外层循环：Q 块 =====
    for start_m in range(0, S, Block_m):
        offs_m = start_m + tl.arange(0, Block_m)          # 当前 Q 块的行号

        q = tl.load(Q + pid_ph * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
                    mask=offs_m[:, None] < S, other=0.0)

        # 三个状态（每个 Q 块重置，所以重复扫KV块，不会污染结果）
        # tl.full：创建一个指定形状，用某个值填满的张量
        m_i = tl.full(shape=(Block_m,), value=float("-inf"), dtype=tl.float32)  
        l_i = tl.zeros((Block_m,), tl.float32)
        acc = tl.zeros((Block_m, D), tl.float32)

        # causal：这个 Q 块只需要扫到「最后一行」的对角块
        # 当前 Q 块从 [start_m, start_m + block_M)，所以 start_m + Block_m 是这个 Q 块最后一行的下一行；hi 就是把它和 S 取个 min
        # 得到 这个 Q 块最多需要扫到的 K/V 位置
        '''
            casual 规则是 offs_m >= offs_n：第 m 行只能看 0..m 这些列，score矩阵的列
            当前 Q 块最后一行是 start_m + Block_m - 1，它能看到的最后一列就是 start_m + Block_m - 1，
            对应的块是 start_n = floor((start_m + Block_m - 1) / Block_n) * Block_n
        '''
        hi = tl.minimum(start_m + Block_m, S)
        # hi 优化跳过的是整块对整块 Q 都无效的 K 块，这是 块级优化

        # ===== 内层循环：K/V 块 =====
        # Q 的第 m 行(query 位置m)能看到的 K 是第0 1 ... m 行(key 位置0 1 ... m)，共 m+1 个 key
        # 是 K 的行，不是 K 的列 ———— K 的列是特征维 D，和 causal 无关
        # 但掩码作用在Score矩阵上，不是K上，K的值没有改变，只是语义上是不让Q看到未来的K的值
        for start_n in range(0, hi, Block_n):
            offs_n = start_n + tl.arange(0, Block_n)

            k = tl.load(K + pid_ph * stride_kh + offs_n[:, None] * stride_km + offs_d[None, :] * stride_kd,
                        mask=offs_n[:, None] < S, other=0.0)
            v = tl.load(V + pid_ph * stride_vh + offs_n[:, None] * stride_vm + offs_d[None, :] * stride_vd,
                        mask=offs_n[:, None] < S, other=0.0)

            s = tl.dot(q, tl.trans(k)) * scale                                      # (Block_m, Block_n)
            # 无效块靠 hi 跳过；对角块上的有效元素靠 tl.where
            s = tl.where(offs_n[None, :] < S, s, float("-inf"))                     # 越界列
            s = tl.where(offs_m[:, None] >= offs_n[None, :], s, float("-inf"))      # causal mask 未来位置不能看

            m_new = tl.maximum(m_i, tl.max(s, 1))
            alpha = tl.exp2((m_i - m_new) * 1.44269504)
            p = tl.exp2((s - m_new[:, None]) * 1.44269504)
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)
            m_i = m_new

        # 归一化写回（每个 Q 块写一次）
        o = acc / l_i[:, None]
        tl.store(O + pid_ph * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
                 o.to(tl.float16), mask=offs_m[:, None] < S)


def flash_attn(q, k, v):
    B, H, S, D = q.shape
    o = torch.empty_like(q)
    Block_m, Block_n = 64, 128
    grid = (B * H,)                        # FA1：只有 (b,h) 一维

    _flash_attn_kernel[grid](q, k, v, o,
                             q.stride(1), q.stride(2), q.stride(3),
                             k.stride(1), k.stride(2), k.stride(3),
                             v.stride(1), v.stride(2), v.stride(3),
                             o.stride(1), o.stride(2), o.stride(3),
                             S, (1.0 / D ** 0.5),
                             Block_m, Block_n, D=D,
                             num_warps=4, num_stages=2)
    return o


if __name__ == "__main__":
    B, H, S, D = 1, 1, 200, 64
    q, k, v = (torch.randn(B, H, S, D, device="cuda", dtype=torch.float16) for _ in range(3))

    score = (q.float() @ k.float().transpose(-1, -2)) * (1.0 / D ** 0.5)
    mask = torch.ones(S, S, device="cuda", dtype=torch.bool).tril()
    ref = torch.softmax(score.masked_fill(~mask, float("-inf")), dim=-1) @ v.float()

    out = flash_attn(q, k, v).float()
    print("关卡B(FA1 调度 + causal):", (out - ref).abs().max().item())     # 期望 ~3e-4
