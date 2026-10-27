"""实验 D（v2）：graph 收益随 N 线性增长？——用 min-of-rounds 压噪声。

为什么改 v2：
    v1 每个 N 只测一轮，数据被系统噪声污染（普通连发在 N=8 和 N=16 之间跳了 3 倍）。
    真实情况里有两件事混在一起：
        ① 每个 kernel 的派发成本（graph 内 ~0.85 µs / 普通 ~9–14 µs）
        ② graph 重放本身的固定开销（launch 一次图，~几十 µs）
    所以正确测法是：每个配置重复多轮取 **最小值**（最小值最接近真实下限），
    并把"整批时间"也打出来，才能看见 ① 和 ② 分别多大。

用法：
    cd ~/projects/aigpu-lab
    python kernels/exp_graph_chain.py
"""

import torch
from torch.utils.cpp_extension import load_inline

CU = "kernels/exp_launch_overhead.cu"
CPP_DECL = r"""
void empty_launch();
void empty_launch_on(int64_t stream_ptr);
void spin_launch(int64_t cycles, int64_t stream_ptr);
void write1_launch(torch::Tensor out);
"""


def build():
    return load_inline(
        name="exp_launch_overhead",
        cpp_sources=CPP_DECL,
        cuda_sources=open(CU, encoding="utf-8").read(),
        functions=["empty_launch", "empty_launch_on", "spin_launch", "write1_launch"],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )


def make_plain_measurer(mod, n):
    def once():
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(n):
            mod.empty_launch()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) * 1e3            # 整批 µs
    return once


def make_graph_measurer(mod, n):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        mod.empty_launch_on(side.cuda_stream)     # 捕获前预热，避免捕获期 malloc
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=side):
        for _ in range(n):
            mod.empty_launch_on(side.cuda_stream)
    torch.cuda.synchronize()

    def once():
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        g.replay()                                 # 一次重放 = n 个 kernel
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) * 1e3             # 整批 µs
    return once


def min_of(fn, rounds=9):
    for _ in range(30):                            # 全局预热
        fn()
    torch.cuda.synchronize()
    return min(fn() for _ in range(rounds))


def main():
    print(f"设备: {torch.cuda.get_device_name(0)}")
    mod = build()

    print()
    print(f"{'N':>4} {'普通整批 µs':>12} {'graph 整批 µs':>14} "
          f"{'普通/个 µs':>11} {'graph/个 µs':>12} {'每层省 µs':>10}")
    print("-" * 74)

    rows = []
    for n in (1, 2, 4, 8, 16, 32, 64):
        p = min_of(make_plain_measurer(mod, n))
        g = min_of(make_graph_measurer(mod, n))
        rows.append((n, p, g))
        print(f"{n:4d} {p:12.1f} {g:14.1f} {p/n:11.2f} {g/n:12.2f} {(p-g)/n:10.2f}")

    # 用最小二乘粗估"graph 重放的固定成本"和"每节点成本"
    ns = [r[0] for r in rows]
    gs = [r[2] for r in rows]
    mean_n = sum(ns) / len(ns)
    mean_g = sum(gs) / len(gs)
    num = sum((n - mean_n) * (g - mean_g) for n, g in zip(ns, gs))
    den = sum((n - mean_n) ** 2 for n in ns)
    slope = num / den                                  # µs / 节点
    intercept = mean_g - slope * mean_n                # µs，固定成本

    print()
    print(f"拟合：graph 整批时间 ≈ {intercept:.1f} µs（固定）+ {slope:.2f} µs × N")
    print(f"→ 重放一张图的固定开销 ≈ {intercept:.0f} µs；图内每个节点 ≈ {slope:.2f} µs")
    if slope < 0:
        print("  （斜率是负的 → 噪声还没压干净，把 rounds 调大再跑一次）")
    print("""
────────────────────────────────────────────────
怎么读：

1. 看 "graph/个" 这一列是否稳定在 0.8~1.0 µs。
   稳 → 图内派发成本与 N 无关，这就是 graph 的单价。

2. 看 "普通/个" 这一列，正常应该在 9~14 µs 之间抖动。
   抖得厉害说明 WSL/后台进程在抢 CPU，但最小值仍然是可信的。

3. N=1 时 graph 通常**更慢**，因为要付固定开销。
   拐点 = 固定开销 ÷ 每个 kernel 省下的 µs。
   你之前测的固定开销 ~37 µs、每个省 ~8.4 µs → 拐点约 N=4~5。

4. 判据（写成式子，面试能直接用）：
       收益 = N × 8.4 µs − 固定开销
       值得用 graph ⇔ 这个值 ÷ 一步总时间 > 5%
────────────────────────────────────────────────
""")


if __name__ == "__main__":
    main()
