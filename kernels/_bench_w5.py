"""w5 flash attention 计时：Triton vs torch SDPA。L2 冲刷 + event 计时，口径与 bench_kernels.py 一致。"""
import importlib.util
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("w5", os.path.join(HERE, "w5_flash_attn.py"))
w5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w5)

PEAK_BW = 272e9          # B/s（标称）
FLUSH_MB = 256


def bench(fn, args, warmup=10, iters=30, flush=True):
    buf = torch.empty(int(FLUSH_MB * 1024 * 1024 // 4), dtype=torch.float32, device="cuda")
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        if flush:
            buf.zero_()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(*args)
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]


def report(tag, ms, B, H, S, D):
    # flash 不把 S² 落显存：DRAM 只走 Q/K/V/O = 4·S·D（K/V 重读命中 L2）
    nbytes = 4 * S * D * B * H * 2
    flops = 4.0 * B * H * S * S * D
    gbps = nbytes / (ms * 1e-3) / 1e9
    tflops = flops / (ms * 1e-3) / 1e12
    print(f"{tag:12s} {ms:8.3f} ms   {gbps:7.1f} GB/s ({gbps * 1e9 / PEAK_BW * 100:5.1f}% peak)"
          f"   {tflops:6.2f} TFLOPS ({tflops / 65.2 * 100:4.1f}% of fp16 peak)")


def main():
    B, H, S, D = 4, 8, 1024, 64
    torch.manual_seed(0)
    q, k, v = (torch.randn(B, H, S, D, device="cuda", dtype=torch.float16) for _ in range(3))

    out = w5.flash_attn(q, k, v)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    print("allclose vs SDPA:", torch.allclose(out.float(), ref.float(), atol=1e-2, rtol=1e-2),
          " max_abs:", (out.float() - ref.float()).abs().max().item())

    t_triton = bench(w5.flash_attn, (q, k, v))
    t_sdpa = bench(torch.nn.functional.scaled_dot_product_attention, (q, k, v))
    report("triton w5", t_triton, B, H, S, D)
    report("torch SDPA", t_sdpa, B, H, S, D)
    print(f"speedup vs SDPA: {t_sdpa / t_triton:.2f}x")
    print("w4 naive 基线 15.53 ms →", f"{15.53 / t_triton:.2f}x faster")


if __name__ == "__main__":
    main()
