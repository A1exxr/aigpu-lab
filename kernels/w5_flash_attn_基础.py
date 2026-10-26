import torch
import triton
import triton.language as tl


@triton.jit
def _attn_cut_kernel(Q, K, V, O,                                    # 指向各自张量数据首元素的“基地址指针”
                     stride_qh, stride_qm, stride_qd,               # 
                     stride_kh, stride_km, stride_kd,
                     stride_vh, stride_vm, stride_vd,
                     stride_oh, stride_om, stride_od,
                     S, D, scale,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):             # BLOCK_M 一个 program 处理多少 Q 行； BLOCK_N 取多少个 key
    # tl.program_id(axis) 返回的是当前程序实例 在指定轴上的索引 ID ———— 和下面包装函数里定义的 grid 元组的元素 是一一对应的关系
    # tl.program_id(0) 对应 pid_m：它返回的是当前实例在第 0 轴上的 ID
    pid_m  = tl.program_id(0)                                                   # 第几个 Q 块

    # tl.program_id(1) 对应 pid_bh：它返回的是当前实例在第 1 轴上的 ID
    pid_bh = tl.program_id(1)                                                   # 第几个 (batch * head) 组合，bh每增加1，就是换一个head，跨越S*D个元素

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)                            # (BLOCK_M,)  query 行号
    # offs_n 从 0 开始，不是从 pid_m * BLOCK_M 开始 ———— 每个 Q 块都只和前 64 个 key 做 attention
    # 完整版应该遍历所有 key
    offs_n = tl.arange(0, BLOCK_N)                                              # (BLOCK_N,)  key 行号（0~63）
    offs_d = tl.arange(0, D)                                                    # (D,)       维度

    '''
        以 pid_bh=0, offs_m[i]=3, offs_d[j]=5 为例，q_ptrs[i,j] 的地址是：
        Q +     0*S*D  +  3*D    +     5*1
        ↑         ↑        ↑            ↑
        基址    跳0个head  跳3行        跳5列
               (0个元素)  (3*64=192)    (5)
            
    '''
# ============================================================================
# Q 的偏移计算 —— 具体例子
# ============================================================================
#
# 假设：
#   B=1, H=1, S=8, D=3（为了手算方便，真实是 S=1024, D=64）
#   Q 形状 (B, H, S, D) = (1, 1, 8, 3)，连续布局
#   Q 在内存里：
#       Q[0,0,0,:] = [q00, q01, q02]   地址 0, 1, 2
#       Q[0,0,1,:] = [q10, q11, q12]   地址 3, 4, 5
#       Q[0,0,2,:] = [q20, q21, q22]   地址 6, 7, 8
#       ...
#       Q[0,0,7,:] = [q70, q71, q72]   地址 21,22,23
#
#   stride_qh = q.stride(1) = S*D = 24      （跳一个 (b,h)）
#   stride_qm = q.stride(2) = D   = 3       （跳一行）
#   stride_qd = q.stride(3) = 1             （跳一个元素）
#
#   本次 program：
#       pid_bh = 0                         （第 0 个 (b,h)）
#       pid_m  = 1                         （第 1 个 Q 块）
#       BLOCK_M = 4                        （一个 program 处理 4 行）
#       D = 3
#
# ----------------------------------------------------------------------------
# ① 行号 offs_m
# ----------------------------------------------------------------------------
#   offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
#          = 1 * 4 + [0, 1, 2, 3]
#          = [4, 5, 6, 7]                   形状 (4,)
#
#   offs_m[:, None] = [[4],
#                      [5],
#                      [6],
#                      [7]]                形状 (4, 1)   ← 列向量
#
# ----------------------------------------------------------------------------
# ② 列号 offs_d
# ----------------------------------------------------------------------------
#   offs_d = tl.arange(0, D)
#          = [0, 1, 2]                     形状 (3,)
#
#   offs_d[None, :] = [[0, 1, 2]]          形状 (1, 3)   ← 行向量
#
# ----------------------------------------------------------------------------
# ③ 行偏移 × stride_qm（stride_qm 是标量 3）
# ----------------------------------------------------------------------------
#   offs_m[:, None] * stride_qm
#   = [[4],
#      [5],
#      [6],
#      [7]] * 3
#   = [[12],
#      [15],
#      [18],
#      [21]]                              形状 (4, 1)
#
#   含义：第 4 行从地址 12 开始，第 5 行从地址 15 开始，...
#
# ----------------------------------------------------------------------------
# ④ 列偏移 × stride_qd（stride_qd 是标量 1）
# ----------------------------------------------------------------------------
#   offs_d[None, :] * stride_qd
#   = [[0, 1, 2]] * 1
#   = [[0, 1, 2]]                          形状 (1, 3)
#
#   含义：第 0 列偏移 0，第 1 列偏移 1，第 2 列偏移 2
#
# ----------------------------------------------------------------------------
# ⑤ 广播相加：(4, 1) + (1, 3) → (4, 3)
# ----------------------------------------------------------------------------
#           [0, 1, 2]      （列偏移，广播成 4 行）
#           -----------
#   [12]    [12, 13, 14]   ← offs_m[0]=4 → 地址 12,13,14
#   [15]    [15, 16, 17]   ← offs_m[1]=5 → 地址 15,16,17
#   [18]    [18, 19, 20]   ← offs_m[2]=6 → 地址 18,19,20
#   [21]    [21, 22, 23]   ← offs_m[3]=7 → 地址 21,22,23
#
# ----------------------------------------------------------------------------
# ⑥ 加上基址和 pid_bh 偏移
# ----------------------------------------------------------------------------
#   q_ptrs = Q + pid_bh*stride_qh + 上面的偏移矩阵
#          = Q + 0*24 + [[12,13,14],
#                        [15,16,17],
#                        [18,19,20],
#                        [21,22,23]]
#
#   所以 q_ptrs[i, j] 指向：
#       q_ptrs[0,0] → Q 地址 12 → Q[0,0,4,0] = q40
#       q_ptrs[0,1] → Q 地址 13 → Q[0,0,4,1] = q41
#       q_ptrs[0,2] → Q 地址 14 → Q[0,0,4,2] = q42
#       q_ptrs[1,0] → Q 地址 15 → Q[0,0,5,0] = q50
#       ...
#       q_ptrs[3,2] → Q 地址 23 → Q[0,0,7,2] = q72
#
#   正好覆盖 Q[0,0,4:8, 0:3]，即第 4~7 行、全部 3 列。
#
# ----------------------------------------------------------------------------
# ⑦ tl.load 一次读整块
# ----------------------------------------------------------------------------
#   q = tl.load(q_ptrs)
#
#   q 的形状 (4, 3)，内容：
#       [[q40, q41, q42],
#        [q50, q51, q52],
#        [q60, q61, q62],
#        [q70, q71, q72]]
#
#   这就是"第 1 个 Q 块"（行 4~7）的完整数据。
#   线程怎么分配、怎么合并访存，全由 Triton 编译器处理。
#
# ============================================================================

    # “Q 块”是 BLOCK_M 行 × D 列。offs_m 定位每一行，offs_d 定位行内每一列，两者广播成 (BLOCK_M, D) 的地址矩阵
    # 为什么“整行都要”仍然需要 offs_d
    # 因为 Triton 不是 C 的指针运算，tl.load 需要一个完整的地址矩阵，每个元素一个地址。
    # 即使 D 维在内存里是连续的，Triton 也不会“自动从行首读到行尾”。必须显式给出：行偏移和列偏移
    q = tl.load(Q + pid_bh*stride_qh + offs_m[:, None]*stride_qm + offs_d[None, :]*stride_qd)
    k = tl.load(K + pid_bh*stride_kh + offs_n[:, None]*stride_km + offs_d[None, :]*stride_kd)
    v = tl.load(V + pid_bh*stride_vh + offs_n[:, None]*stride_vm + offs_d[None, :]*stride_vd)

    s = tl.dot(q, tl.trans(k)) * scale                  # (BLOCK_M, BLOCK_N)  算分数
    m = tl.max(s, 1)                                    # (BLOCK_M,)  行最大值，每行一个
    p = tl.exp2((s - m[:, None]) * 1.44269504)          # * 1.44269504：log2(e)，把 exp 转成 exp2；
    l = tl.sum(p, 1)                                    # (BLOCK_M,) 行和，每行一个l
    acc = tl.dot(p.to(tl.float16), v)                   # (BLOCK_M, D)  加权求和； p 转回 fp16，因为 Tensor Core 的 fp16 输入快
    o = acc / l[:, None]                                # 完成 softmax 归一化； l[:, None] 广播成 (BLOCK_M, 1)；  结果 (BLOCK_M, D)

    tl.store(O + pid_bh*stride_oh + offs_m[:, None]*stride_om + offs_d[None, :]*stride_od,
             o.to(tl.float16))


def attn_cut(q, k, v):
    B, H, S, D = q.shape
    o = torch.empty_like(q)
    BLOCK_M, BLOCK_N = 64, 64                   # 截断版：只看前 64 个 key
    grid = (triton.cdiv(S, BLOCK_M), B * H)     # 第一维：Q 块数 = S / BLOCK_M； 第二维：(b,h) 总数；  没有第三维，也没有 K/V 循环
    _attn_cut_kernel[grid](q, k, v, o,
                           # 这里只传了一个参数，但 * 可以拆包，把 q.stride() 返回的 stride 元组“拆开”，逐个作为位置参数传给函数
                           # 对于张量 q， .stride() 返回一个元组，长度等于张量的维度数，每个元素是该维度上跨一步需要跳过的元素个数 
                           # 返回(H*S*D, S*D, D, 1)， 跳一个 batch，跨 H*S*D 个元素 ———— 跳一个 head，跨 S*D 个元素 ———— 跳一行，跨 D 个元素 ———— 跳一个元素，跨 1 个元素
                           # kernel里只需要3个参数，而这里和kernel里也没有reshape，所以会错
                           *q.stride(), *k.stride(), *v.stride(), *o.stride(),          
                           S, D, 1.0 / (D ** 0.5),
                           BLOCK_M, BLOCK_N, num_warps=4)
    return o



if __name__ == "__main__":
    B, H, S, D = 1, 1, 128, 64
    q, k, v = (torch.randn(B, H, S, D, device="cuda", dtype=torch.float16) for _ in range(3))
    out = attn_cut(q, k, v)
    ref = torch.softmax((q @ k[:, :, :64].transpose(-1, -2)) * (1/8), dim=-1) @ v[:, :, :64]
    print(torch.allclose(out.float(), ref.float(), atol=1e-2, rtol=1e-2), out.shape)