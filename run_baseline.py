"""W1 交付物：测量 PyTorch 原生算子的基线性能。

用法（在 WSL 里，项目目录下）：
    python run_baseline.py
    python run_baseline.py --seq 4096 --dtype fp16 --iters 300
    python run_baseline.py --out results/baseline.md

它会产出：
    1. 你的显卡规格 + Roofline 拐点
    2. 每个算子的：中位数延迟 / P99 / 有效带宽 / 带宽利用率 / 实测算力 / Roofline 判定
    3. attention 的「标准实现 vs torch SDPA」对比（含峰值显存）—— 这是你 FlashAttention 项目的动机
    4. 一份 markdown 报告，直接可以贴进 README

**W1 的验收标准：你能独立跑出这些数字，并解释为什么要 warmup、为什么要 flush L2。**
"""
from __future__ import annotations

import argparse
import math

import torch
import torch.nn.functional as F

from gpulab import report, specs, timer, verify, workloads

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def peak_mem_mb(fn) -> float:
    """测一次调用的峰值显存占用（MB）。FlashAttention 的显存优势靠这个量化。"""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024 ** 2


def rms_norm(x, weight, eps: float = 1e-6):
    """RMSNorm 的 PyTorch 朴素实现（多次读写 HBM，正是你要优化的对象）。"""
    var = x.float().pow(2).mean(dim=-1, keepdim=True)
    return (x.float() * torch.rsqrt(var + eps)).to(x.dtype) * weight


# ---------------------------------------------------------------------------
# 构造被测算子
# ---------------------------------------------------------------------------

def build_ops(args, dtype):
    B, S, H = args.bs, args.seq, args.hidden
    NH, DH = args.heads, args.head_dim
    dev = "cuda"
    eb = workloads.elem_bytes(dtype)
    n_tokens = B * S * H

    x = torch.randn(B, S, H, device=dev, dtype=dtype)
    z = torch.randn(B, S, H, device=dev, dtype=dtype)
    w = torch.randn(H, device=dev, dtype=dtype)
    b = torch.randn(H, device=dev, dtype=dtype)

    M = K = N = args.matmul_size
    a = torch.randn(M, K, device=dev, dtype=dtype)
    bm = torch.randn(K, N, device=dev, dtype=dtype)

    q = torch.randn(B, NH, S, DH, device=dev, dtype=dtype)
    k = torch.randn(B, NH, S, DH, device=dev, dtype=dtype)
    v = torch.randn(B, NH, S, DH, device=dev, dtype=dtype)
    scale = 1.0 / math.sqrt(DH)

    def attn_naive(q=q, k=k, v=v, scale=scale):
        """标准实现：S 和 P 两个 S×S 矩阵会实实在在落到 HBM 里。"""
        scores = (q @ k.transpose(-2, -1)) * scale
        probs = F.softmax(scores, dim=-1)
        return probs @ v

    ops = [
        dict(
            group="elementwise / norm (memory-bound 预期)",
            name="residual_add",
            detail=f"({B},{S},{H})",
            nbytes=workloads.bytes_elementwise((B, S, H), dtype, 2, 1),
            flops=float(n_tokens),
            fn=lambda x=x, z=z: torch.add(x, z),
        ),
        dict(
            group="elementwise / norm (memory-bound 预期)",
            name="gelu",
            detail=f"({B},{S},{H})",
            nbytes=workloads.bytes_elementwise((B, S, H), dtype, 1, 1),
            flops=8.0 * n_tokens,
            fn=lambda x=x: F.gelu(x),
        ),
        dict(
            group="elementwise / norm (memory-bound 预期)",
            name="softmax(dim=-1)",
            detail=f"({B},{S},{H})",
            nbytes=workloads.bytes_softmax((B, S, H), dtype),
            flops=5.0 * n_tokens,
            fn=lambda x=x: F.softmax(x, dim=-1),
        ),
        dict(
            group="elementwise / norm (memory-bound 预期)",
            name="layer_norm",
            detail=f"({B},{S},{H})",
            nbytes=workloads.bytes_layer_norm((B, S, H), dtype),
            flops=8.0 * n_tokens,
            fn=lambda x=x, w=w, b=b: F.layer_norm(x, (H,), w, b),
        ),
        dict(
            group="elementwise / norm (memory-bound 预期)",
            name="rms_norm (朴素实现)",
            detail=f"({B},{S},{H})",
            nbytes=workloads.bytes_layer_norm((B, S, H), dtype, affine=False) + H * eb,
            flops=6.0 * n_tokens,
            fn=lambda x=x, w=w: rms_norm(x, w),
        ),
        dict(
            group="GEMM (compute-bound 预期)",
            name="matmul",
            detail=f"({M},{K})@({K},{N})",
            nbytes=workloads.bytes_matmul(M, N, K, dtype),
            flops=workloads.flops_matmul(M, N, K),
            fn=lambda a=a, bm=bm: a @ bm,
        ),
        dict(
            group="attention (本项目的重点)",
            name="attn_naive (materialize)",
            detail=f"B={B} H={NH} S={S} D={DH}",
            nbytes=workloads.attention_core(B, NH, S, DH, dtype, materialize_scores=True)[1],
            flops=workloads.attention_core(B, NH, S, DH, dtype, materialize_scores=True)[0],
            fn=attn_naive,
        ),
        dict(
            group="attention (本项目的重点)",
            name="attn_sdpa (torch 融合)",
            detail=f"B={B} H={NH} S={S} D={DH}",
            nbytes=workloads.attention_core(B, NH, S, DH, dtype, materialize_scores=False)[1],
            flops=workloads.attention_core(B, NH, S, DH, dtype, materialize_scores=False)[0],
            fn=lambda q=q, k=k, v=v: F.scaled_dot_product_attention(q, k, v),
        ),
    ]
    return ops


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="PyTorch 原生算子基线测量")
    ap.add_argument("--bs", type=int, default=1, help="batch size")
    ap.add_argument("--seq", type=int, default=2048, help="序列长度")
    ap.add_argument("--hidden", type=int, default=4096, help="hidden 维度")
    ap.add_argument("--heads", type=int, default=8, help="attention head 数")
    ap.add_argument("--head-dim", type=int, default=64, help="每个 head 的维度")
    ap.add_argument("--matmul-size", type=int, default=4096, help="GEMM 的 M=N=K")
    ap.add_argument("--dtype", default="fp16", choices=list(DTYPES))
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--no-flush-l2", action="store_true",
                    help="关闭 L2 冲刷（默认开启；关掉会让带宽虚高，仅用于对比演示）")
    ap.add_argument("--out", default="results/baseline.md")
    args = ap.parse_args()

    dtype = DTYPES[args.dtype]
    spec = specs.detect()

    print("=" * 78)
    print(spec)
    print("=" * 78)
    print(f"dtype={args.dtype}  warmup={args.warmup}  iters={args.iters}  "
          f"flush_l2={not args.no_flush_l2}")
    print("=" * 78)

    ops = build_ops(args, dtype)
    rows, last_group = [], None

    for op in ops:
        if op["group"] != last_group:
            print(f"\n--- {op['group']} ---")
            last_group = op["group"]

        stats = timer.bench_report(
            op["name"], op["fn"],
            warmup=args.warmup, iters=args.iters, flush_l2=not args.no_flush_l2,
        )

        ms = stats["median_ms"]
        bw = workloads.bandwidth_gbps(op["nbytes"], ms)
        tf = workloads.achieved_tflops(op["flops"], ms)
        bound = spec.classify(op["flops"], op["nbytes"], args.dtype)

        rows.append([
            op["group"].split(" (")[0],
            op["name"],
            op["detail"],
            op["nbytes"] / 1024 ** 2,
            op["flops"] / 1e9,
            ms,
            stats["p99_ms"],
            bw,
            100.0 * bw / spec.bw_gbps,
            tf,
            100.0 * tf / spec.peak_tflops(args.dtype),
            bound,
        ])

    # ---- attention 显存对比（FlashAttention 的动机）----
    print("\n--- attention 峰值显存对比 ---")
    mem_rows = []
    for op in ops:
        if "attn" not in op["name"]:
            continue
        peak = peak_mem_mb(op["fn"])
        naive_peak = workloads.attention_naive_nbytes(
            args.bs, args.heads, args.seq, args.head_dim, dtype) / 1024 ** 2
        mem_rows.append([op["name"], peak, naive_peak])
        print(f"{op['name']:<28} 实测峰值={peak:9.1f} MB   "
              f"理论 S/P 矩阵={naive_peak:9.1f} MB")

    # ---- 生成报告 ----
    headers = ["类别", "算子", "shape", "访存(MB)", "FLOPs(G)", "中位数(ms)",
               "P99(ms)", "有效带宽(GB/s)", "带宽利用率(%)", "实测算力(TFLOPS)",
               "算力利用率(%)", "Roofline 判定"]
    table = report.md_table(headers, rows)
    mem_table = report.md_table(["attention 实现", "实测峰值显存(MB)",
                                 "理论 S/P 矩阵(MB)"], mem_rows)

    sections = [
        report.env_block(spec, extra=f"参数: bs={args.bs} seq={args.seq} "
                                     f"hidden={args.hidden} heads={args.heads} "
                                     f"head_dim={args.head_dim} dtype={args.dtype}"),
        "## 基线性能\n", table, "",
        "> 有效带宽 = 访存字节数 / 中位数延迟；利用率相对本卡峰值 "
        f"{spec.bw_gbps:.1f} GB/s 计算。\n"
        "> 访存字节数按 `gpulab/workloads.py` 中的公式估算，假设已写入该文件。\n",
        "## Attention 显存对比\n", mem_table, "",
        "## 结论\n",
        "（先跑出数字，再回来写结论。至少回答：）\n",
        "1. 哪些算子是 memory-bound，哪些是 compute-bound？和 Roofline 预测一致吗？",
        "2. attention 的 `attn_naive` 和 `attn_sdpa` 差距有多大？差在哪（延迟 / 显存）？",
        "3. 你的 memory-bound 算子带宽利用率到了多少？离峰值还差多少？",
        "4. 关闭 L2 冲刷（`--no-flush-l2`）重跑，带宽数字变成什么样？为什么？",
    ]

    report.write_report(args.out, "PyTorch 原生算子基线（W1）", sections)


if __name__ == "__main__":
    main()
