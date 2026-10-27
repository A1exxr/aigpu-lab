"""实验 A：一次 kernel 启动到底要多久（launch overhead）。

用法（WSL）：
    cd ~/projects/aigpu-lab
    python kernels/exp_launch_overhead.py

这个 .py 里没有 kernel，它只做三件事：
    1. 把同目录的 exp_launch_overhead.cu 读成一个字符串
    2. 交给 load_inline → 用 nvcc 编译成一个 Python 模块
    3. 调这个模块里的函数，计时
原理和你的 bench_kernels.py:715 build_module() 完全一样。
"""

import time

import torch
from torch.utils.cpp_extension import load_inline

CU = "kernels/exp_launch_overhead.cu"

# 告诉编译器：这个 .cu 里对外暴露哪两个函数（和 .cu 里的 host wrapper 签名一致）
CPP_DECL = r"""
void empty_launch();
void write1_launch(torch::Tensor out);
"""


def build():
    src = open(CU, encoding="utf-8").read()      # 读 .cu 源码
    return load_inline(
        name="exp_launch_overhead",              # 换这个名字会重新编译一次
        cpp_sources=CPP_DECL,
        cuda_sources=src,
        functions=["empty_launch", "write1_launch"],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )


def measure_submit_cost(fn, n=2000):
    """只测 CPU 提交耗时：不 sync，所以测到的是"塞进队列"的成本。"""
    for _ in range(100):          # 预热：首次调用触发模块加载/上下文初始化
        fn()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    t_submit = (time.perf_counter() - t0) / n * 1e6      # µs/次，纯 CPU 侧

    torch.cuda.synchronize()      # 排空队列
    return t_submit


def measure_total_cost(fn, n=2000):
    """提交 + 执行 + 排空 的总时间。"""
    for _ in range(100):
        fn()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e6


def measure_gpu_only(fn, n=2000):
    """CUDA event 测 GPU 侧：一个 event 包住 n 次启动。"""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for _ in range(100):
        fn()
    torch.cuda.synchronize()

    start.record()
    for _ in range(n):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n * 1e3              # µs/次


def main():
    if not torch.cuda.is_available():
        raise SystemExit("没有 CUDA 设备")
    print(f"设备: {torch.cuda.get_device_name(0)}")

    mod = build()
    out = torch.zeros(1, device="cuda", dtype=torch.float32)

    rows = []
    for name, fn in [
        ("空 kernel   <<<1,32>>>", mod.empty_launch),
        ("写 1 个元素 <<<1,32>>>", lambda: mod.write1_launch(out)),
    ]:
        rows.append((name, measure_submit_cost(fn), measure_total_cost(fn), measure_gpu_only(fn)))

    print()
    print(f"{'kernel':26s} {'CPU提交 µs':>10s} {'总均摊 µs':>10s} {'GPU侧 µs':>10s}")
    print("-" * 62)
    for name, s, t, g in rows:
        print(f"{name:26s} {s:10.2f} {t:10.2f} {g:10.2f}")

    torch.cuda.synchronize()
    print(f"\nwrite1 之后 out[0] = {out.item():.0f}（应该是 2100，说明 kernel 真的跑了）")

    print("""
────────────────────────────────────────────────
跑完先自己回答这三问，再贴给我：

1. 空 kernel 的 "CPU提交" 是多少 µs？这个数字代表什么？
2. "GPU侧" 和 "总均摊" 差在哪？为什么 GPU侧 明显更小？
3. 你的 PagedAttention decode 是 1.04 ms。用第 1 题的数字算：
   一次启动的固定开销占 kernel 时间的百分之几？
   → 如果 < 1%，你**不是** launch-bound，实验 B/C 的收益上限就被卡住了。
────────────────────────────────────────────────
""")


if __name__ == "__main__":
    main()
