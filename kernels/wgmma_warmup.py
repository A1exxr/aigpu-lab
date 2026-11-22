"""
wgmma_warmup.py —— D5 单测的 driver：验 wgmma 描述符语义（LBO/SBO）与累加器映射。

形状：C[64][64] = A[64][16] @ B[16][64]，fp16 输入 / fp32 累加，1 个 block = 128 线程 = 1 个 warpgroup。
需要 sm_90a（Hopper）：H100 / H800 / H200 / H20 都行，H20 算力低但语义一致。

用法（在仓库根目录）：
    python kernels/wgmma_warmup.py
"""
import sys
from pathlib import Path

import torch

CU_PATH = Path(__file__).resolve().parent / "wgmma_warmup.cu"

# 这个 .cu 里只有 kernel、没有 host 入口，所以在这里「拼一个」host 包装再交给 load_inline。
# 不修改原文件：只把文本拼在内存里编译。
HOST_WRAPPER = r"""
// ---------------- 由 wgmma_warmup.py 追加的 host 入口 ----------------
#include <torch/extension.h>

torch::Tensor wgmma_warmup(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "A/B 必须在 GPU 上");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A/B 必须是 2D");
    TORCH_CHECK(A.scalar_type() == torch::kFloat16, "A 必须是 fp16");
    TORCH_CHECK(B.scalar_type() == torch::kFloat16, "B 必须是 fp16");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "A/B 必须连续");
    TORCH_CHECK(A.size(0) == M && A.size(1) == K, "A 必须是 [64,16]");
    TORCH_CHECK(B.size(0) == K && B.size(1) == N, "B 必须是 [16,64]");

    auto C = torch::empty({M, N}, A.options().dtype(torch::kFloat32));
    wgmma_warmup_kernel<<<1, NTHREADS>>>(
        reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(B.data_ptr<at::Half>()),
        C.data_ptr<float>());
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "launch 失败: ", cudaGetErrorString(err));
    return C;
}
"""

CPP_DECL = "torch::Tensor wgmma_warmup(torch::Tensor A, torch::Tensor B);"


def check_arch():
    if not torch.cuda.is_available():
        sys.exit("没有 CUDA 设备")
    cc = torch.cuda.get_device_capability()
    name = torch.cuda.get_device_name()
    print(f"GPU: {name}  compute_cap = {cc[0]}.{cc[1]}")
    if cc[0] < 9:
        sys.exit(f"需要 Hopper（compute_cap 9.x），当前 {cc[0]}.{cc[1]} —— 这个 kernel 在别处跑不了")
    return cc


def main():
    cc = check_arch()
    src = CU_PATH.read_text(encoding="utf-8")
    for need in ("wgmma_warmup_kernel", "#define M", "#define K", "#define N", "#define NTHREADS"):
        assert need in src, f"{CU_PATH.name} 里找不到 {need}"

    from torch.utils.cpp_extension import load_inline

    print("编译中（首次约 30–60 秒）...")
    mod = load_inline(
        name="wgmma_warmup_ext",
        cpp_sources=CPP_DECL,
        cuda_sources=src + HOST_WRAPPER,
        functions=["wgmma_warmup"],
        extra_cuda_cflags=[f"-arch=sm_{cc[0]}{cc[1]}a", "-O3"],
        verbose=False,
    )

    torch.manual_seed(0)
    A = torch.randn(64, 16, device="cuda", dtype=torch.float16)
    B = torch.randn(16, 64, device="cuda", dtype=torch.float16)

    C = mod.wgmma_warmup(A, B)
    ref = (A.float() @ B.float())
    diff = (C - ref).abs().max().item()
    rel = diff / ref.abs().max().item()

    print(f"C shape = {tuple(C.shape)}  max|diff| = {diff:.6e}  相对 = {rel:.3e}")
    ok = diff <= 2e-2
    print("PASS" if ok else "FAIL")
    if not ok:
        print("\n失败定位提示：")
        print("  * 若整体错乱（数值量级对、位置错） → A 的 core-matrix 重排或 a_lbo/a_sbo 错")
        print("  * 若只有右半边错 → B 的块序应为 n 外层（b_lbo=128 / b_sbo=256 那个配置）")
        print("  * 若数值全 0 → wgmma_fence / commit / wait0 顺序不对")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
