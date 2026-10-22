"""w5 flash attention 调参扫描：Block_m / Block_n / num_warps / num_stages。

不改 w5_flash_attn.py，直接用它里面的 _flash_attn_kernel 重新发起不同 launch 配置。
用法：~/miniconda3/envs/triton/bin/python _tune_w5.py
"""
import importlib.util
import os

import torch
import triton

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("w5", os.path.join(HERE, "w5_flash_attn.py"))
w5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w5)

B, H, S, D = 4, 8, 1024, 64
SCALE = 1.0 / D ** 0.5
FLUSH_MB = 256

torch.manual_seed(0)
q, k, v = (torch.randn(B, H, S, D, device="cuda", dtype=torch.float16) for _ in range(3))
ref = torch.nn.functional.scaled_dot_product_attention(q, k, v).float()
o = torch.empty_like(q)
flush_buf = torch.empty(int(FLUSH_MB * 1024 * 1024 // 4), dtype=torch.float32, device="cuda")


def run_once(Block_m, Block_n, warps, stages):
    grid = (triton.cdiv(S, Block_m), B * H)
    w5._flash_attn_kernel[grid](
        q, k, v, o,
        q.stride(1), q.stride(2), q.stride(3),
        k.stride(1), k.stride(2), k.stride(3),
        v.stride(1), v.stride(2), v.stride(3),
        o.stride(1), o.stride(2), o.stride(3),
        S, SCALE, Block_m, Block_n,
        D=D, num_warps=warps, num_stages=stages,
    )


def bench(Block_m, Block_n, warps, stages, warmup=10, iters=20):
    for _ in range(warmup):
        run_once(Block_m, Block_n, warps, stages)
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        flush_buf.zero_()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        run_once(Block_m, Block_n, warps, stages)
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]


def main():
    # 基线：SDPA
    def sdpa():
        return torch.nn.functional.scaled_dot_product_attention(q, k, v)

    for _ in range(10):
        sdpa()
    torch.cuda.synchronize()
    ts = []
    for _ in range(20):
        flush_buf.zero_()
        s0, e0 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s0.record()
        sdpa()
        e0.record()
        e0.synchronize()
        ts.append(s0.elapsed_time(e0))
    ts.sort()
    sdpa_ms = ts[len(ts) // 2]
    print(f"torch SDPA 基线: {sdpa_ms:.3f} ms\n")

    rows = []
    for Block_m in (64, 128):
        for Block_n in (64, 128):
            for warps in (4, 8):
                for stages in (1, 2, 3):
                    tag = f"M{Block_m}/N{Block_n}/w{warps}/s{stages}"
                    try:
                        run_once(Block_m, Block_n, warps, stages)
                        torch.cuda.synchronize()
                        err = (o.float() - ref).abs().max().item()
                        if err > 1e-2:
                            print(f"{tag:22s} 结果错 max_abs={err:.4f}")
                            continue
                        ms = bench(Block_m, Block_n, warps, stages)
                        rows.append((ms, tag, err))
                        print(f"{tag:22s} {ms:7.3f} ms   {sdpa_ms / ms:5.2f}x SDPA   max_abs={err:.5f}")
                    except Exception as exc:                       # 编译/资源超限
                        print(f"{tag:22s} FAILED: {type(exc).__name__}: {str(exc)[:80]}")

    rows.sort()
    print("\n按时间排序（前 5）:")
    for ms, tag, _ in rows[:5]:
        print(f"  {tag:22s} {ms:7.3f} ms   {sdpa_ms / ms:5.2f}x SDPA   {8.59e9 / (ms * 1e-3) / 1e12:6.2f} TFLOPS")
    if rows:
        best_ms, best_tag, _ = rows[0]
        print(f"\n最优: {best_tag}  {best_ms:.3f} ms  "
              f"(原配置 M64/N64/w4/s1 见上表)")


if __name__ == "__main__":
    main()
