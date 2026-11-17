"""
w12_fa2.py —— w12_fa2.cu 的 driver：正确性（对 SDPA）+ 性能

跑法：
    cd ~/projects/aigpu-lab
    ~/miniconda3/envs/triton/bin/python kernels/w12_fa2.py

判据（两档）：
    1) S=512, B=2, H=8, Hkv=2, D=64   -> max diff <= 3e-3  PASS
    2) S=125（tail 不满一块）同 shape   -> max diff <= 3e-3  PASS

口径说明：
    - 参考实现用 fp32 的 F.scaled_dot_product_attention(is_causal=True)
    - TFLOPS 按因果算：flops = 2 * B*H*S*(S+1)*D（QK 与 PV 各一半的三角）
    - SDPA 基线用 expand 之后的 KV 头（GQA 展开成 MHA），这是最公平的对照
"""

import math
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

HERE = Path(__file__).resolve().parent
CU = (HERE / "w12_fa2.cu").read_text(encoding="utf-8")

CPP = "torch::Tensor fa2_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, double scale);"

print("[build] compiling w12_fa2 ... (第一次 40-70 s，之后再跑走缓存)", flush=True)
mod = load_inline(
    name="w12_fa2",
    cpp_sources=CPP,
    cuda_sources=CU,
    functions=["fa2_forward"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
    verbose=False,
)
print("[build] ok", flush=True)


def make_inputs(B, H, Hkv, S, D, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    # 缩放小一点，避免 exp 溢出把误差放大
    q = (torch.randn(B, H, S, D, generator=g, device="cuda") * 0.5).half()
    k = (torch.randn(B, Hkv, S, D, generator=g, device="cuda") * 0.5).half()
    v = (torch.randn(B, Hkv, S, D, generator=g, device="cuda") * 0.5).half()
    return q, k, v


def ref_fp32(q, k, v, H, Hkv, scale):
    grp = H // Hkv
    kk = k.float().repeat_interleave(grp, dim=1)
    vv = v.float().repeat_interleave(grp, dim=1)
    return F.scaled_dot_product_attention(q.float(), kk, vv, is_causal=True, scale=scale)


def check(B, H, Hkv, S, D, atol=3e-3):
    scale = 1.0 / math.sqrt(D)
    q, k, v = make_inputs(B, H, Hkv, S, D)
    out = mod.fa2_forward(q, k, v, scale)
    torch.cuda.synchronize()

    ref = ref_fp32(q, k, v, H, Hkv, scale)
    diff = (out.float() - ref).abs()
    md = diff.max().item()
    ok = md <= atol
    print(f"  S={S:<5} out{tuple(out.shape)}  max|diff|={md:.3e}  "
          f"ref_absmax={ref.abs().max().item():.3f}  {'PASS' if ok else 'FAIL'}")
    return ok, scale, q, k, v, H, Hkv


def bench(B, H, Hkv, S, D, warmup=10, iters=50):
    scale = 1.0 / math.sqrt(D)
    q, k, v = make_inputs(B, H, Hkv, S, D, seed=1)
    grp = H // Hkv
    kk = k.repeat_interleave(grp, dim=1).contiguous()
    vv = v.repeat_interleave(grp, dim=1).contiguous()

    def timeit(fn):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            st, en = torch.cuda.Event(True), torch.cuda.Event(True)
            st.record()
            fn()
            en.record()
            torch.cuda.synchronize()
            ts.append(st.elapsed_time(en))
        return statistics.median(ts)

    t_ours = timeit(lambda: mod.fa2_forward(q, k, v, scale))
    t_ref = timeit(lambda: F.scaled_dot_product_attention(
        q, kk, vv, is_causal=True, scale=scale))

    flops = 2.0 * B * H * S * (S + 1) * D       # QK + PV，三角
    tf_ours = flops / (t_ours * 1e-3) / 1e12
    tf_ref = flops / (t_ref * 1e-3) / 1e12
    print(f"  B={B} H={H} Hkv={Hkv} S={S} D={D}")
    print(f"  ours : {t_ours:8.3f} ms   {tf_ours:6.2f} TFLOPS")
    print(f"  SDPA : {t_ref:8.3f} ms   {tf_ref:6.2f} TFLOPS")
    print(f"  ratio: {t_ref / t_ours:.2f}x  (>1 表示我们更快)")
    return t_ours, t_ref


def main():
    if not torch.cuda.is_available():
        print("no CUDA"); return 1
    print(f"[env] {torch.cuda.get_device_name(0)}  torch {torch.__version__}")

    print("\n[1] 正确性")
    ok1, *_ = check(2, 8, 2, 512, 64)
    ok2, *_ = check(2, 8, 2, 125, 64)
    ok = ok1 and ok2

    print("\n[2] 性能（fp16，cuBLAS/SDPA 同 shape 对照）")
    bench(2, 8, 2, 512, 64)
    bench(2, 8, 2, 1024, 64)

    print("\n[3] 结论")
    print("  PASS" if ok else "  FAIL —— 先修正确性，别看性能")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
