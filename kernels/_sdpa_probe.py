"""核查 SDPA 的公平基线：后端是谁、GQA 怎么处理、各自多快。

用法（WSL）：
    export PATH=$HOME/miniconda3/envs/triton/bin:$PATH
    python kernels/_sdpa_probe.py
"""
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

torch.manual_seed(0)
B, H, HKV, S, D = 2, 8, 2, 1024, 64
dev, dt = "cuda", torch.float16

q = torch.randn(B, H, S, D, device=dev, dtype=dt)
k = torch.randn(B, HKV, S, D, device=dev, dtype=dt)
v = torch.randn(B, HKV, S, D, device=dev, dtype=dt)

flops = 2 * B * H * S * (S + 1) * D  # causal 口径


def bench(fn, iters=30, warmup=10, rounds=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    best = []
    for _ in range(rounds):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        best.append(s.elapsed_time(e) / iters)
    best.sort()
    return best[len(best) // 2]


def report(tag, fn):
    try:
        ms = bench(fn)
        print(f"{tag:38s} {ms:8.4f} ms   {flops / ms / 1e9:8.2f} TFLOPS")
    except Exception as ex:
        print(f"{tag:38s} FAILED: {type(ex).__name__}: {str(ex)[:80]}")


ke = k.repeat_interleave(H // HKV, dim=1)   # 老写法：先展开，再当普通 MHA
ve = v.repeat_interleave(H // HKV, dim=1)

print(f"shape B={B} H={H} Hkv={HKV} S={S} D={D} dtype={dt}")
print(f"flops(causal) = {flops/1e9:.2f} GFLOP\n")

report("math  backend (fp32 里算)", lambda: F.scaled_dot_product_attention(
    q.float(), ke.float(), ve.float(), is_causal=True).to(dt))
report("mem_efficient (repeat_interleave)", lambda: F.scaled_dot_product_attention(
    q, ke, ve, is_causal=True))
report("flash (repeat_interleave)", lambda: F.scaled_dot_product_attention(
    q, ke, ve, is_causal=True))
report("flash + enable_gqa=True", lambda: F.scaled_dot_product_attention(
    q, k, v, is_causal=True, enable_gqa=True))

# 后端自动选择时到底用了谁
from torch.nn.attention import sdpa_kernel as _sk
for name, be in [("FLASH", SDPBackend.FLASH_ATTENTION),
                 ("EFFICIENT", SDPBackend.EFFICIENT_ATTENTION),
                 ("CUDNN", SDPBackend.CUDNN_ATTENTION),
                 ("MATH", SDPBackend.MATH)]:
    try:
        with _sk(be):
            F.scaled_dot_product_attention(q, ke, ve, is_causal=True)
            torch.cuda.synchronize()
        print(f"backend {name:10s} 可用")
    except Exception as ex:
        print(f"backend {name:10s} 不可用: {type(ex).__name__}: {str(ex)[:60]}")

# 用 profiler 看默认路径真实的 kernel 名
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    F.scaled_dot_product_attention(q, ke, ve, is_causal=True)
    torch.cuda.synchronize()
kernels = [e.key for e in prof.key_averages() if e.device_type == torch.autograd.DeviceType.CUDA]
print("\n默认路径用到的 kernel（前 6 个）：")
for kk in kernels[:6]:
    print("   ", kk)
