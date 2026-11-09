"""探针：`paged_attn_split` 与 SDPA 在 mini 引擎真实 shape 下差多少（单测 kernel，不走模型）

背景：M3b-2 接上后整机反而慢（S=13 时 0.86×、S=441 时 0.40×），而 Python 侧
开销已排除（228 次调用 CPU 累计仅 9.7 ms，占 258 ms 的 4%）。
这个脚本只回答一个问题：**同一个 shape 下，kernel 与 SDPA 各多少毫秒。**

跑：
    cd ~/projects/aigpu-lab
    ~/miniconda3/envs/triton/bin/python engines/_probe_attn_shapes.py
"""

import pathlib

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

from m3_paged import BLOCK_SIZE, N_BLOCKS

KERNEL = pathlib.Path(__file__).resolve().parent.parent / "kernels" / "w7_paged_attn_split.cu"
DECL = """torch::Tensor paged_attn_split(torch::Tensor q, torch::Tensor k_cache,
                                  torch::Tensor v_cache, torch::Tensor block_table,
                                  torch::Tensor context_lens);"""
H, D, NB = 12, 64, N_BLOCKS
REP = 200


def build():
    return load_inline(name="probe_paged_split", cpp_sources=DECL,
                       cuda_sources=KERNEL.read_text(encoding="utf-8"),
                       functions=["paged_attn_split"],
                       extra_cuda_cflags=["-O3", "--use_fast_math"], verbose=False)


def bench(fn, rep=REP):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(rep):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / rep


def main():
    mod = build()
    dev, dt = "cuda", torch.float16
    k = torch.randn(NB, BLOCK_SIZE, H, D, device=dev, dtype=dt)
    v = torch.randn(NB, BLOCK_SIZE, H, D, device=dev, dtype=dt)
    scaling = 1.0 / D ** 0.5
    print(f"H={H} D={D} BLOCK_SIZE={BLOCK_SIZE}  （kernel 按 H=32,D=128,L=2048 设计）\n")
    print(f"{'n':>5} | {'kernel ms':>10} | {'SDPA ms':>9} | {'kernel/SDPA':>11}")
    for n in (13, 128, 450, 1000):
        nb = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
        table = torch.arange(nb, device=dev, dtype=torch.int32).unsqueeze(0)
        lens = torch.tensor([n], device=dev, dtype=torch.int32)
        q = (torch.randn(1, H, 1, D, device=dev, dtype=dt) * scaling).squeeze(2).contiguous()
        # SDPA 的对照：把前 n 个 token 从池里拼出来（HF 的 cache 就是连续张量）
        kk = k[:nb].reshape(-1, H, D)[:n].transpose(0, 1).unsqueeze(0).contiguous()
        vv = v[:nb].reshape(-1, H, D)[:n].transpose(0, 1).unsqueeze(0).contiguous()
        t_k = bench(lambda: mod.paged_attn_split(q, k, v, table, lens))
        t_s = bench(lambda: F.scaled_dot_product_attention(
            q.unsqueeze(2), kk, vv, scale=1.0))
        print(f"{n:>5} | {t_k:>10.4f} | {t_s:>9.4f} | {t_k/t_s:>10.1f}x")


if __name__ == "__main__":
    main()
