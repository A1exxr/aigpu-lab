"""实验 C：CUDA Graph 到底省了多少？（拿真数字，不靠嘴说）

用法（WSL）：
    cd ~/projects/aigpu-lab
    python kernels/exp_graph_capture.py

期望看到：普通调用 ~9 µs/次，graph 重放 ~1 µs/次。差 5~10 倍就是 graph 的全部价值。

会踩的坑（先知道，省得怀疑代码）：
    - 必须用非默认流捕获，所以 .cu 里多了 empty_launch_on(stream_ptr)
    - 捕获期间不能有 cudaMalloc / 不能有同步 / 不能读 GPU 结果
    - graph 重放时参数是"快照"，我们这里 kernel 没有参数，所以安全
"""

import time

import torch
from torch.utils.cpp_extension import load_inline

CU = "kernels/exp_launch_overhead.cu"
CPP_DECL = r"""
void empty_launch();
void empty_launch_on(int64_t stream_ptr);
void write1_launch(torch::Tensor out);
"""


def build():
    return load_inline(
        name="exp_launch_overhead",
        cpp_sources=CPP_DECL,
        cuda_sources=open(CU, encoding="utf-8").read(),
        functions=["empty_launch", "empty_launch_on", "write1_launch"],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )


def time_plain_launches(mod, n=2000):
    """普通方式：n 次启动，CUDA event 包住整体。"""
    for _ in range(100):
        mod.empty_launch()
    torch.cuda.synchronize()

    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n):
        mod.empty_launch()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / n * 1e3          # µs/次


def time_graph_replay(mod, n_kernels=1000, n_replay=20):
    """把 n_kernels 个空 kernel 录进一张图，然后重放 n_replay 次。"""
    side = torch.cuda.Stream()                  # 非默认流（关键！）
    side.wait_stream(torch.cuda.current_stream())

    # 先跑一次，确保所有初始化都完成（捕获期间不允许 malloc）
    with torch.cuda.stream(side):
        mod.empty_launch_on(side.cuda_stream)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=side):
        for _ in range(n_kernels):
            mod.empty_launch_on(side.cuda_stream)   # 只录，不执行
    torch.cuda.synchronize()

    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n_replay):
        g.replay()
    e.record()
    torch.cuda.synchronize()

    per_replay_us = s.elapsed_time(e) / n_replay * 1e3
    return per_replay_us / n_kernels, per_replay_us


def main():
    print(f"设备: {torch.cuda.get_device_name(0)}")
    mod = build()

    plain = time_plain_launches(mod)
    per_kernel_in_graph, per_replay = time_graph_replay(mod)

    print()
    print(f"{'方式':22s} {'每次启动 µs':>14s} {'每次 replay µs':>16s}")
    print("-" * 58)
    print(f"{'普通 <<<>>> 启动':22s} {plain:14.2f} {'-':>16s}")
    print(f"{'1000 kernel 的 graph':22s} {per_kernel_in_graph:14.2f} {per_replay:16.2f}")
    print()
    print(f"节省倍数: {plain / per_kernel_in_graph:.2f}x")

    print("""
────────────────────────────────────────────────
两问（这回答案我直接给，你只需确认数字对不对）：

1. 为什么 graph 里每个 kernel 只要 ~1 µs？
   因为 graph 把"1000 次 kernel 提交"打包成 1 次驱动调用，CPU 不再逐条写队列。

2. 那为什么你的 decode 不用它？
   因为 decode 一次 1.04 ms，省 8 µs 只有 0.8%。
   但如果一个 kernel 只有 5 µs（真实 LLM 每 token 上百个小 kernel），
   8 µs 的节省 > kernel 本身，graph 就从"可选"变成"必须"。
────────────────────────────────────────────────
""")


if __name__ == "__main__":
    main()
