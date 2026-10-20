"""_perf_attn.py —— 不用 ncu，直接用 torch.profiler 量 w4 三个 kernel 各自的耗时。

用法（在 WSL 的 kernels 目录下）：
    python _perf_attn.py

它复用 bench_kernels.py 里的 REGISTRY / build_module，所以形状、dtype、
输入构造和跑分时完全一致，不需要重复搭环境。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

# 让 `import bench_kernels` 能找到 gpulab（bench 脚本自己也做了这件事）
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

# REGISTRY 里的 cuda_file 是相对仓库根的（"kernels/w4_attn_naive.cu"），
# 所以必须先切到仓库根，否则从 kernels/ 目录跑会报“文件不存在”。
os.chdir(HERE.parent)

import bench_kernels as B  # noqa: E402


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "attn_naive"
    spec = next((s for s in B.REGISTRY if s.name == name), None)
    if spec is None:
        raise SystemExit(f"REGISTRY 里没有 {name}")

    case = spec.cases[0]
    print(f"[build] {spec.name}  <-  {spec.cuda_file}")
    mod, err = B.build_module(spec, verbose=False)   # 返回 (module, error)
    if err:
        raise SystemExit(err)

    dev = "cuda"
    dtype = B.DTYPES[case.dtype]
    inputs = case.make_inputs(case.shape, dtype, dev)

    # 预热 3 次（含 cuBLAS / 分配器 / JIT 的首轮开销）
    for _ in range(3):
        case.cuda_call(mod, inputs)
    torch.cuda.synchronize()

    iters = 5
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
    ) as prof:
        for _ in range(iters):
            case.cuda_call(mod, inputs)
        torch.cuda.synchronize()

    print(f"\n=== 每个 kernel 的 CUDA 时间（{iters} 次调用的总和）===\n")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=12))

    print("\n=== 折算到单次调用 ===")
    total_us = 0.0
    for evt in prof.key_averages():
        # torch 2.9+ 把 cuda_time_total 改名成 self_device_time_total
        us = getattr(evt, "self_device_time_total", None)
        if us is None:
            us = getattr(evt, "cuda_time_total", 0.0) or 0.0
        if us <= 0:
            continue
        per_call = us / iters
        total_us += per_call
        if per_call >= 1.0:
            print(f"  {evt.key[:48]:<50} {per_call:10.1f} µs")
    print(f"  {'合计':<48} {total_us:10.1f} µs")


if __name__ == "__main__":
    main()
