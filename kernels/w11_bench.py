"""w11 反向性能对照：前向 / 台阶2（atomic）/ 台阶3（无 atomic）/ 台阶4（fp16）

用法（在 kernels/ 目录下，因为各模块互相 import）：
    cd ~/projects/aigpu-lab/kernels
    ~/miniconda3/envs/triton/bin/python w11_bench.py

每个实现跑 3 轮取最小，避免冷启动污染。
"""

import torch
import triton
import triton.testing

from w11_bwd_2_chunked import attn_bwd_triton_chunked
from w11_bwd_3_split import attn_bwd_triton_split
from w11_bwd_4_fp16 import attn_bwd_triton_split_fp16
from w11_oracle import bwd_ref, fwd, make_inputs

B, H, S, D = 1, 8, 1024, 64


def bench(fn, rounds=3):
    return min(triton.testing.do_bench(fn, warmup=25, rep=100) for _ in range(rounds))


def main():
    q32, k32, v32, do32 = make_inputs(B, H, S, D, seed=3, dtype=torch.float32)
    s_f, p_f, o32, lse32 = fwd(q32, k32, v32)
    dq_ref, dk_ref, dv_ref, *_ = bwd_ref(q32, k32, v32, do32)
    delta32 = (do32 * o32).sum(dim=-1)

    q16, k16, v16, do16 = (t.half() for t in (q32, k32, v32, do32))

    # ---- 先确认三者结果一致（口径不同，判据不同）----
    dq2, dk2, dv2, _ = attn_bwd_triton_chunked(q32, k32, v32, o32, do32, lse32)
    dq3, dk3, dv3 = attn_bwd_triton_split(q32, k32, v32, do32, lse32, delta32)
    dq4, dk4, dv4 = attn_bwd_triton_split_fp16(q16, k16, v16, do16, lse32, delta32)

    print("正确性（台阶2/3 判据 1e-5，台阶4 判据 2e-3）")
    for tag, (a, b, c), thr in (
        ("台阶2", (dq2, dk2, dv2), 1e-5),
        ("台阶3", (dq3, dk3, dv3), 1e-5),
        ("台阶4", (dq4.float(), dk4.float(), dv4.float()), 2e-3),
    ):
        e = max((a - dq_ref).abs().max().item(),
                (b - dk_ref).abs().max().item(),
                (c - dv_ref).abs().max().item())
        print(f"  {tag}: max_abs = {e:.3e}   {'OK' if e < thr else 'FAIL'}  (判据 {thr:g})")

    dq3b, dk3b, dv3b = attn_bwd_triton_split(q32, k32, v32, do32, lse32, delta32)
    same = torch.equal(dq3, dq3b) and torch.equal(dk3, dk3b) and torch.equal(dv3, dv3b)
    print(f"  台阶3 连跑两次逐位相同: {'OK' if same else 'FAIL'}（台阶2 用了 atomic，做不到）")

    # ---- 计时 ----
    t_fwd = bench(lambda: fwd(q32, k32, v32))
    t2 = bench(lambda: attn_bwd_triton_chunked(q32, k32, v32, o32, do32, lse32))
    t3 = bench(lambda: attn_bwd_triton_split(q32, k32, v32, do32, lse32, delta32))
    t4 = bench(lambda: attn_bwd_triton_split_fp16(q16, k16, v16, do16, lse32, delta32))

    flops = 6.0 * B * H * S * S * D      # 粗口径（实际执行 7 个 S²D matmul）

    print()
    print("=" * 70)
    print(f"{'实现':<24}{'ms':>9}{'相对前向':>11}{'相对台阶3':>11}{'TFLOPS':>10}")
    print("-" * 70)
    rows = [
        ("前向（PyTorch fp32）", t_fwd),
        ("台阶2（atomic）", t2),
        ("台阶3（无 atomic）", t3),
        ("台阶4（fp16+TC）", t4),
    ]
    for name, t in rows:
        print(f"{name:<24}{t:>9.3f}{t/t_fwd:>11.2f}{t/t3:>11.2f}{flops/t/1e9:>10.2f}")
    print("=" * 70)
    print(f"fp16 提速 = {t3/t4:.2f}×   （本卡 fp16 峰值 ≈ 65.2 TFLOPS）")
    print(f"台阶4 = {t4:.3f} ms       （目标线 ≤ 1.5 ms）")
    print()
    print("注：前向是 PyTorch 手写 fp32 版（物化 S²），不是 SDPA —— 不能当对外基线。")


if __name__ == "__main__":
    main()
