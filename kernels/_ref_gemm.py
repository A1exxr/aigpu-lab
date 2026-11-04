"""cuBLAS 参考基线：PyTorch matmul 在 1024³ 上的真实速度。

用法（WSL）：
    cd ~/projects/aigpu-lab
    ~/miniconda3/envs/triton/bin/python kernels/_ref_gemm.py

计时方法与我们自己的 kernel 一致：
    预热 200 次 → 5 轮 × 200 次 → 每轮取平均 → 5 轮里取最小值
（单次测量会被升降频干扰，最小值最接近“无干扰”那次）
"""

import torch

M = N = K = 1024
FLOPS = 2.0 * M * N * K          # 2.147e9 FLOP
PEAK_FP16 = 65.2                 # TFLOPS，本卡实测持续频率下的 fp16 Tensor Core 峰值
PEAK_FP32 = 16.3                 # TFLOPS，24 SM × 128 lane × 2 × 2.655 GHz

# PyTorch 默认在 fp32 下可能用 TF32（10 位尾数），这里关掉，保证是“真 fp32”
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def bench(fn, warmup=200, rounds=5, iters=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    best = float("inf")
    for _ in range(rounds):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(iters):
            fn()
        b.record()
        b.synchronize()
        best = min(best, a.elapsed_time(b) / iters)
    return best


def run(dtype, label):
    a = torch.randn(M, K, device="cuda", dtype=dtype)
    b = torch.randn(K, N, device="cuda", dtype=dtype)
    ms = bench(lambda: a @ b)
    tflops = FLOPS / (ms * 1e-3) / 1e12
    peak = PEAK_FP16 if dtype == torch.float16 else PEAK_FP32
    print(f"{label:<28} {ms:8.4f} ms   {tflops:6.2f} TFLOPS   {tflops / peak * 100:5.1f}% of peak")
    return ms


if __name__ == "__main__":
    print(f"形状 M=N=K={M}，FLOP = {FLOPS / 1e9:.3f} G")
    print(f"设备：{torch.cuda.get_device_name(0)}")
    print("-" * 74)

    ms_fp16 = run(torch.float16, "cuBLAS fp16 (Tensor Core)")
    ms_fp32 = run(torch.float32, "cuBLAS fp32 (CUDA core)")

    print("-" * 74)
    OUR_TC = 0.2836      # 台阶4 wmma（之前测的）
    OUR_REG = 0.6574     # 台阶3 寄存器分块
    print(f"{'我们的 台阶4 (wmma)':<28} {OUR_TC:8.4f} ms   "
          f"{FLOPS / (OUR_TC * 1e-3) / 1e12:6.2f} TFLOPS   "
          f"{FLOPS / (OUR_TC * 1e-3) / 1e12 / PEAK_FP16 * 100:5.1f}% of peak")
    print()
    print(f"我们的台阶4 / cuBLAS fp16  = {OUR_TC / ms_fp16:.2f}× 倍的差距")
    print(f"我们的台阶4 / cuBLAS fp32  = {ms_fp32 / OUR_TC:.2f}×  <- 之前用的这句：“超 fp32 基线”")
