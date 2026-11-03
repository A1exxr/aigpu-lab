"""w11 台阶 3 的性能对照：前向 / 台阶2（atomic）/ 台阶3（无 atomic）

用法：
    cd ~/projects/aigpu-lab
    ~/miniconda3/envs/triton/bin/python kernels/w11_bench.py

输出三行：每次跑三遍取最小，避免第一次冷启动污染。
"""

import torch
import triton
import triton.testing

from w11_attn_bwd import attn_bwd_triton_chunked, attn_bwd_triton_split
from w11_oracle import fwd, make_inputs

B, H, S, D = 1, 8, 1024, 64

q, k, v, do = make_inputs(B, H, S, D, seed=3, dtype=torch.float32)
s_f, p_f, o, lse = fwd(q, k, v)
delta = (do * o).sum(dim=-1)

# 正确性先确认一次（别拿错的结果去测速）
dq2, dk2, dv2, _ = attn_bwd_triton_chunked(q, k, v, o, do, lse)
dq3, dk3, dv3 = attn_bwd_triton_split(q, k, v, do, lse, delta)
print("台阶2 vs 台阶3 的差（应 ~1e-7，但不会逐位相同）:")
print(f"  dq {(dq2-dq3).abs().max().item():.3e}  dk {(dk2-dk3).abs().max().item():.3e}"
      f"  dv {(dv2-dv3).abs().max().item():.3e}")


def bench(fn, rounds=3):
    return min(triton.testing.do_bench(fn, warmup=25, rep=100) for _ in range(rounds))


t_fwd = bench(lambda: fwd(q, k, v))
t_s2 = bench(lambda: attn_bwd_triton_chunked(q, k, v, o, do, lse))
t_s3 = bench(lambda: attn_bwd_triton_split(q, k, v, do, lse, delta))

print()
print("=" * 64)
print(f"{'实现':<28}{'ms':>10}{'相对前向':>12}{'相对台阶2':>12}")
print("-" * 64)
print(f"{'前向 fwd':<28}{t_fwd:>10.3f}{1.0:>12.2f}{'-':>12}")
print(f"{'台阶2（atomic）':<28}{t_s2:>10.3f}{t_s2/t_fwd:>12.2f}{1.0:>12.2f}")
print(f"{'台阶3（无 atomic）':<28}{t_s3:>10.3f}{t_s3/t_fwd:>12.2f}{t_s3/t_s2:>12.2f}")
print("=" * 64)
print()
print(f"台阶3 / 前向 = {t_s3/t_fwd:.2f}×   （台阶4 的验收线是 ≤ 3×、≤1.5 ms）")
print(f"台阶3 / 台阶2 = {t_s3/t_s2:.2f}×   （多算一遍 p/ds 的代价，预期 1.1–1.5×）")
