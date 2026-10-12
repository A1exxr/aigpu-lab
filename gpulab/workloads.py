"""各类算子的 FLOPs / 访存字节数计算公式。

**为什么要有这个文件：**
    "有效带宽" = 访存字节数 / 时间。字节数必须你自己算出来。
    没有字节数，就没有带宽利用率，就没有 Roofline 分析，
    你的性能报告就只剩一个光秃秃的延迟数字 —— 那没有说服力。

**重要原则：所有公式都带假设，写完必须把假设写进报告。**
    面试官问"你这个字节数怎么算的"，你要能一条条讲出来。
"""
from __future__ import annotations

import torch

# ---------------------------------------------------------------------------
# 基础
# ---------------------------------------------------------------------------

_DTYPE_BYTES = {
    torch.float32: 4,
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.int8: 1,
    torch.float64: 8,
}


def elem_bytes(dtype: torch.dtype) -> int:
    """单个元素占多少字节。"""
    if dtype in _DTYPE_BYTES:
        return _DTYPE_BYTES[dtype]
    # 兜底：用 torch 自己算
    return torch.empty(1, dtype=dtype).element_size()


def numel(shape) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


# ---------------------------------------------------------------------------
# 访存字节数
# ---------------------------------------------------------------------------

def bytes_elementwise(shape, dtype, n_read: int = 1, n_write: int = 1) -> int:
    """逐元素算子（add / gelu / relu / dropout 等）。

    假设：每个输入读一次，每个输出写一次。没有中间缓冲。
    """
    return numel(shape) * elem_bytes(dtype) * (n_read + n_write)


def bytes_layer_norm(shape, dtype, affine: bool = True) -> int:
    """LayerNorm / RMSNorm（沿最后一维归一化）。

    假设：这是**最小访存**的融合实现 —— 读输入一次、写输出一次。

    ⚠️ 重要限制：如果你测的是**多趟（unfused）实现**，真实访存会显著高于本公式。
       例如 PyTorch 里这种写法：

           var = x.float().pow(2).mean(dim=-1, keepdim=True)
           return (x.float() * torch.rsqrt(var + eps)).to(x.dtype) * weight

       `x.float()` 会物化一个 fp32 中间张量（2 倍大小），`pow(2)`、`mean`、乘法
       各自都要读写 HBM。实测中这种实现的真实访存是最小值的十倍量级，
       但本公式仍只报最小值 → **算出来的"带宽利用率"会严重偏低**。

       实测案例：朴素 RMSNorm 在 (1,2048,4096) fp16 上耗时 1.378 ms，
       按本公式算只有 9% 带宽利用率。但那 9% 是**公式低估**造成的，
       不是 kernel 真的慢十倍。真实结论应该是："这个实现有大量冗余访存"。

       处理办法：要么改用真实字节数（自己算中间张量），要么在报告里注明
       "此处的带宽利用率是下界估计"。**不要拿一个被低估的分母去下结论。**
    """
    eb = elem_bytes(dtype)
    n = numel(shape)
    last = shape[-1]
    total = n * eb * 2  # 读输入 + 写输出
    if affine:
        total += 2 * last * eb
    return total


def bytes_softmax(shape, dtype) -> int:
    """Softmax（沿最后一维）。读一次、写一次。"""
    return numel(shape) * elem_bytes(dtype) * 2


def bytes_matmul(M: int, N: int, K: int, dtype) -> int:
    """GEMM: (M,K) @ (K,N) -> (M,N)。三个矩阵各读写一次。"""
    eb = elem_bytes(dtype)
    return (M * K + K * N + M * N) * eb


# ---------------------------------------------------------------------------
# FLOPs
# ---------------------------------------------------------------------------

def flops_matmul(M: int, N: int, K: int, bias: bool = False) -> float:
    """GEMM 的浮点运算次数 = 2*M*N*K（每次乘加算 2 次）。"""
    f = 2.0 * M * N * K
    if bias:
        f += M * N
    return f


# ---------------------------------------------------------------------------
# Attention（这是你项目的重点，公式要能讲清楚）
# ---------------------------------------------------------------------------

def attention_core(B: int, H: int, S: int, D: int, dtype,
                   causal: bool = False,
                   materialize_scores: bool = True) -> tuple[float, int]:
    """标准 attention 核心部分的 (FLOPs, 访存字节数)。

    只算 attention 核心（Q@K^T → softmax → P@V），不含 QKV 投影和输出投影。

    参数
    ----
    B, H, S, D : batch / head 数 / 序列长度 / head 维度
    causal     : 因果掩码（下三角），FLOPs 减半
    materialize_scores : True = 标准实现（S 和 P 两个 S×S 矩阵落到 HBM）
                         False = FlashAttention 式融合（中间矩阵不进 HBM）

    FLOPs 假设
    ----------
        Q@K^T : 2*B*H*S*S*D
        P@V   : 2*B*H*S*S*D
        合计 4*B*H*S*S*D（causal 时乘 0.5）

    访存假设
    --------
        共同部分：读 Q,K,V（3*B*H*S*D）+ 写 O（B*H*S*D）
        materialize_scores=True 时额外增加：
            S 矩阵：写一次 + 读一次（softmax）
            P 矩阵：写一次 + 读一次（P@V）
            即 4 * B*H*S*S 个元素

    ⚠️ 这是**保守估计**，真实实现里 S/P 可能被读写更多次。
       报告里必须写明你采用的是这套假设。
    """
    eb = elem_bytes(dtype)
    flops = 4.0 * B * H * S * S * D
    if causal:
        flops *= 0.5

    nbytes = (3 * B * H * S * D) * eb      # 读 Q, K, V
    nbytes += (B * H * S * D) * eb         # 写 O

    if materialize_scores:
        nbytes += 4 * B * H * S * S * eb   # S/P 的写+读

    return flops, nbytes


def attention_naive_nbytes(B: int, H: int, S: int, D: int, dtype) -> int:
    """标准 attention 的**显存占用**（不是流量），用于算 O(S²) 的峰值。

    S 矩阵 + P 矩阵：2 * B*H*S*S 个元素。
    FlashAttention 下这一项是 O(S)，所以长序列时显存差距非常戏剧化 ——
    这正是你要在 4060 Ti（显存只有 8G 或 16G）上展示的效果。
    """
    return 2 * B * H * S * S * elem_bytes(dtype)


def attention_flash_nbytes(B: int, H: int, S: int, D: int, dtype,
                           block: int = 128) -> int:
    """FlashAttention 的**显存占用**：只保留 O(S*block) 的统计量。

        m (running max) 和 l (running sum)：各 B*H*S 个 fp32
        输出 O 分块：O(block) 级别，可忽略
    """
    return 2 * B * H * S * 4


# ---------------------------------------------------------------------------
# 便捷：算利用率
# ---------------------------------------------------------------------------

def bandwidth_gbps(nbytes: int, median_ms: float) -> float:
    """有效带宽 (GB/s)。"""
    if median_ms <= 0:
        return float("inf")
    return nbytes / (median_ms * 1e-3) / 1e9


def achieved_tflops(flops: float, median_ms: float) -> float:
    """实测算力 (TFLOPS)。"""
    if median_ms <= 0:
        return float("inf")
    return flops / (median_ms * 1e-3) / 1e12
