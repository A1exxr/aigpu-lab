"""w6_3gemm 的 driver：正确性 vs cuBLAS + 交替测速（丢首轮、取中位数）。"""
import statistics
import sys
from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline

CU = (Path(__file__).resolve().parent / "w6_3gemm.cu").read_text(encoding="utf-8")

CPP = """
#include <torch/extension.h>
torch::Tensor gemm(torch::Tensor A, torch::Tensor B);
"""

EXT = load_inline(
    name="w6_3gemm",
    cpp_sources=CPP,
    cuda_sources=CU,
    functions=["gemm"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=False,
)

M = N = K = 1024
ROUNDS = 6
ITERS = 50


def tflops(ms: float) -> float:
    return 2 * M * N * K / (ms * 1e-3) / 1e12


def main() -> int:
    torch.manual_seed(0)
    dev = "cuda"
    A = torch.randn(M, K, device=dev, dtype=torch.half)
    B = torch.randn(K, N, device=dev, dtype=torch.half)
    cache = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.int32, device=dev)

    C = EXT.gemm(A, B)
    ref = (A.float() @ B.float()).half()
    diff = (C.float() - ref.float()).abs().max().item()
    rel = diff / ref.float().abs().max().item()
    ok = torch.allclose(C, ref, atol=2e-2, rtol=2e-2)
    print(f"max|diff| = {diff:.4e}  相对 = {rel:.2e}  allclose = {ok}")
    if not ok:
        i = (C.float() - ref.float()).abs().argmax().item()
        print("最差点 idx =", i, " ours =", C.flatten()[i].item(), " ref =", ref.flatten()[i].item())
        return 1

    # 交替 A/B，且**每轮交换先后顺序**（否则后跑的一方总是吃到更高的时钟，实测差 20%+）
    ours_ms, cublas_ms = [], []

    def time_ours():
        cache.zero_()
        ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
        ev0.record()
        for _ in range(ITERS):
            EXT.gemm(A, B)
        ev1.record()
        torch.cuda.synchronize()
        return ev0.elapsed_time(ev1) / ITERS

    def time_cublas():
        cache.zero_()
        ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
        ev0.record()
        for _ in range(ITERS):
            A @ B
        ev1.record()
        torch.cuda.synchronize()
        return ev0.elapsed_time(ev1) / ITERS

    for r in range(ROUNDS):
        if r % 2 == 0:
            o = time_ours(); c = time_cublas()
        else:
            c = time_cublas(); o = time_ours()
        ours_ms.append(o); cublas_ms.append(c)
        print(f"  round {r}: ours {o:.4f} ms   cuBLAS {c:.4f} ms   ({'ours 先跑' if r % 2 == 0 else 'cuBLAS 先跑'})")

    o = statistics.median(ours_ms[1:])
    c = statistics.median(cublas_ms[1:])
    print(f"\n{'':<10}{'ms':>10}{'TFLOPS':>10}")
    print(f"{'ours':<10}{o:>10.4f}{tflops(o):>10.2f}")
    print(f"{'cuBLAS':<10}{c:>10.4f}{tflops(c):>10.2f}")
    print(f"\n占 cuBLAS = {c / o * 100:.1f}%   比值 = {c / o:.3f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
