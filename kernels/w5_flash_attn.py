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
                       Block_m:tl.constexpr, Block_n:tl.constexpr, D:tl.constexpr):     # 这里得把D设置为编译期常量， 因为offs_d = tl.arange(0, D)，tl.arange()参数得是编译器常量
    pid_m = tl.program_id(0)            
    pid_ph = tl.program_id(1)

    offs_m = pid_m * Block_m + tl.arange(0, Block_m)        # 当前 Q 块的行号
    offs_d = tl.arange(0, D)                                # # (D,)        所有特征维

    # 加载 Q 块 ———— 基址 + pid_bh 个 (b,h) + 行偏移 + 列偏移
    q = tl.load(Q + pid_ph * stride_qh + offs_m[:, None] * stride_qm + offs_d[None,:] * stride_qd, mask=offs_m[:, None] < S, other=0.0)

    # 初始化三个状态
    m_i = tl.full(shape=(Block_m,), value=float("-inf"), dtype=tl.float32)      # 到目前为止的最大值
    l_i = tl.zeros((Block_m,), tl.float32)                                      # 每行 exp 和
    acc = tl.zeros((Block_m, D), tl.float32)                                    # 每行输出累加器

    # K/V循环
    for start_n in range(0, S, Block_n):                # 串行循环：每个 program 都要走完所有 K/V 块，因为 attention 需要看全部 key
        offs_n = start_n  + tl.arange(0, Block_n)       # 当前K的行号

        k = tl.load(K + pid_ph * stride_kh + offs_n[:, None] * stride_km + offs_d[None, :] * stride_kd, mask = offs_n[:, None] < S, other=0.0)       # (BLOCK_N, D)
        # 这里加载 V 和 K 是一样的，虽然最后是 P @ V，是P的行与V的列进行点积，但V是要被全部访问的，且是行主序且连续，所以还是按行来读取的
        v = tl.load(V + pid_ph * stride_vh + offs_n[:, None] * stride_vm + offs_d[None, :] * stride_vd, mask = offs_n[:, None] < S, other=0.0)       # (BLOCK_N, D)

        # 计算分数
        s = tl.dot(q, tl.trans(k)) * scale              # (BLOCK_M, BLOCK_N)    tl.dot 会自动用 Tensor Core（fp16 输入、fp32 累加），比逐元素乘加快得多
        # 越界处理
        # k 加载时，越界的行被 other=0.0 填成 0；  q @ 0 = 0，这些越界位置会得到一个假的分数 0，如果不管它，exp(0 - m) 会参与 l 和 acc 的计算，污染结果
        # 处理 S 不是 BLOCK_N 整数倍的关键
        s = tl.where(offs_n[None,:] < S, s, float("-inf"))

        # 累加 online softmax版本
        m_new = tl.maximum  (m_i, tl.max(s,1))                      # 新的全局最大值
        alpha = tl.exp2((m_i - m_new) * 1.44269504)                 # 旧尺度 ——> 新尺度
        p = tl.exp2((s - m_new[:, None]) *  1.44269504)             # 当前块权重，用新基准计算当前块权重
        # 注意，这里l_i 形状是 (Block_m,)，和alpha一样，不用广播
        l_i = l_i * alpha + tl.sum(p, 1)                            # 更新分母，旧的exp和 换算到新基准，再加上当前块的贡献
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)    # 更新分子
        m_i = m_new                                                 # 最后才更新基准




    # 归一化写回，在for循环外面，只需要写一次
    o = acc / l_i[:, None]
    tl.store(O + pid_ph * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od, 
                o.to(tl.float16), mask=offs_m[:, None] < S)





def flash_attn(q, k, v):
    B, H, S, D = q.shape
    o = torch.empty_like(q)
    Block_m, Block_n = 64, 128
    grid = (triton.cdiv(S, Block_m), B * H)         # 每个 program 负责：一个 Q 块 × 一个 (b,h)

    _flash_attn_kernel[grid](q, k, v, o,
                             q.stride(1), q.stride(2), q.stride(3),
                             k.stride(1), k.stride(2), k.stride(3),
                             v.stride(1), v.stride(2), v.stride(3),
                             o.stride(1), o.stride(2), o.stride(3),
                             S, (1.0 / D ** 0.5),
                             Block_m, Block_n, D=D,     #  D 作为 constexpr 关键字传入
                             num_warps=4, num_stages=2)     
    return o






if __name__ == "__main__":
    B, H, S, D = 1, 1, 128, 64
    q, k, v = (torch.randn(B, H, S, D, device="cuda", dtype=torch.float16) for _ in range(3))
    ref = torch.softmax((q.float() @ k.float().transpose(-1, -2)) * (1 / 8), dim=-1) @ v.float()
    out = flash_attn(q, k, v).float()
    print("关卡B:", (out - ref).abs().max().item())     # 期望 ~3e-4


