"""_ncu_once.py —— 给 ncu 用的最小目标：只跑一次 attn_naive。

为什么需要它：
    torch.profiler（_perf_attn.py 用的）和 ncu 都通过 CUPTI 采集，
    两者同时开会互相卡住，表现就是「ncu 一直跑不完」。
    所以 profile 时用这个脚本，别用 _perf_attn.py。

用法（在 WSL）：
    export PATH=/usr/local/cuda-12.8/bin:$PATH
    cd ~/projects/aigpu-lab

    # 正确的 ncu 调用（跳过 1 次预热，只抓 1 次 qk_kernel）
    ncu --clock-control none --kernel-name regex:qk_kernel \
        --launch-skip 1 --launch-count 1 --set basic \
        python kernels/_ncu_once.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

# REGISTRY 里的 cuda_file 是相对仓库根的
os.chdir(HERE.parent)

import bench_kernels as B  # noqa: E402


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "attn_naive"
    spec = next((s for s in B.REGISTRY if s.name == name), None)
    if spec is None:
        raise SystemExit(f"REGISTRY 里没有 {name}")

    case = spec.cases[0]
    print(f"[build] {spec.name}  <-  {spec.cuda_file}")
    mod, err = B.build_module(spec, verbose=False)
    if err:
        raise SystemExit(err)

    inp = case.make_inputs(case.shape, B.DTYPES[case.dtype], B.DEV)

    # 预热 1 次（ncu 用 --launch-skip 1 跳过它）
    case.cuda_call(mod, inp)
    torch.cuda.synchronize()

    # 被 profile 的那一次
    case.cuda_call(mod, inp)
    torch.cuda.synchronize()

    print("done")


if __name__ == "__main__":
    main()
