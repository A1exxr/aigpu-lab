"""
w12d_fa2_tile128.py —— w12d_fa2_tile128.cu（TILE_M 64→128、SM_THREADS 128→256）的 driver

跑法：
    cd ~/projects/aigpu-lab
    ~/miniconda3/envs/triton/bin/python kernels/w12d_fa2_tile128.py

判据：
    [1] 三档 max|diff| 必须与 w12c 完全相同：3.644e-04 / 4.944e-04 / 5.429e-04
        （纯 tile 尺寸变化，数值不该有任何漂移；变了就是 row_base 或 smem 偏移错了）
    [2] S=1024 ≥ 10 TFLOPS（w12c = 8.79，0.245 ms）
    [3] S=125 这一档：TILE_M=128 > S，必须先在 kernel/driver 里处理
        TODO D1：决定 pad 到 128（F.pad 补零行）还是 S<128 时退回 w12c 的 kernel

TFLOPS 口径同 w12c：flops = 2 * B * H * S * (S+1) * D
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

HERE = Path(__file__).resolve().parent
CU = (HERE / "w12d_fa2_tile128.cu").read_text(encoding="utf-8")

CPP = "torch::Tensor fa2_mma_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, double scale);"

print("[build] compiling w12d_fa2_tile128 ... (第一次 60-90 s)", flush=True)
mod = load_inline(
    name="w12d_fa2_tile128",
    cpp_sources=CPP,
    cuda_sources=CU,
    functions=["fa2_mma_forward"],
    extra_cuda_cflags=["-O3", "-lineinfo"],
    verbose=False,
)
print("[build] ok", flush=True)


def ref_forward(q, k, v, scale, grp):
    """PyTorch 参考：GQA 展开 + 显式 causal mask + 全程 fp32。"""
    kk = k.float().repeat_interleave(grp, dim=1)
    vv = v.float().repeat_interleave(grp, dim=1)
    qq = q.float()
    S = qq.size(-2)
    s = (qq @ kk.transpose(-1, -2)) * scale
    mask = torch.ones(S, S, device=qq.device, dtype=torch.bool).tril()
    s = s.masked_fill(~mask, float("-inf"))
    p = torch.softmax(s, dim=-1)
    return p @ vv


def bench(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
    ev0.record()
    for _ in range(iters):
        fn()
    ev1.record()
    torch.cuda.synchronize()
    return ev0.elapsed_time(ev1) / iters


def run_case(B, H, Hkv, S, D, iters=20):
    torch.manual_seed(0)
    q = (torch.randn(B, H, S, D, device="cuda") * 0.5).half()
    k = (torch.randn(B, Hkv, S, D, device="cuda") * 0.5).half()
    v = (torch.randn(B, Hkv, S, D, device="cuda") * 0.5).half()

    scale = 1.0 / (D ** 0.5)
    grp = H // Hkv

    out = mod.fa2_mma_forward(q, k, v, scale)
    ref = ref_forward(q, k, v, scale, grp)
    mx = (out.float() - ref).abs().max().item()
    ok = mx <= 3e-3

    print(f"  S={S:5d}  max|diff|={mx:.3e}  ref_absmax={ref.abs().max().item():.3f}  "
          f"{'PASS' if ok else 'FAIL'}")

    if S >= 512:
        kk = k.repeat_interleave(grp, dim=1)
        vv = v.repeat_interleave(grp, dim=1)
        t_ours = bench(lambda: mod.fa2_mma_forward(q, k, v, scale), iters)
        t_sdpa = bench(lambda: F.scaled_dot_product_attention(q, kk, vv, is_causal=True), iters)
        flops = 2.0 * B * H * S * (S + 1) * D
        print(f"         ours {t_ours:.3f} ms / {flops / t_ours / 1e9:.2f} TFLOPS"
              f"   ;  SDPA {t_sdpa:.3f} ms / {flops / t_sdpa / 1e9:.2f} TFLOPS"
              f"   ratio {t_sdpa / t_ours:.2f}x")
    return ok


def main():
    print("\n[1] 正确性 + [2] 性能")
    print("  S=125 是尾块用例（125/64 除不尽，且 125 < 2*64，会走 row_base 夹取那条路）")
    B, H, Hkv, D = 2, 8, 2, 64
    ok1 = run_case(B, H, Hkv, 512, D)
    ok2 = run_case(B, H, Hkv, 125, D)
    ok3 = run_case(2, 8, 2, 1024, D)

    print("\n[3] 结论")
    if ok1 and ok2 and ok3:
        print("  PASS —— causal + GQA 的 FA2 前向（mma.sync 版）跑通")
        print("  下一步：① 看 S=1024 的 TFLOPS 是否 >= 4（低于 4 就别急着做 v3）")
        print("          ② 再加 split-K / 寄存器分块；③ 然后才是 v3（TMA + wgmma）")
        return 0
    print("  FAIL —— 先看是哪一档：S=512 挂 → 查 softmax/PV；只有 S=125 挂 → 查行尾/块夹取")
    return 1


if __name__ == "__main__":
    sys.exit(main())
