"""实验 B：多 stream 到底能不能让 kernel 同时跑？（真数字）

用法（WSL）：
    cd ~/projects/aigpu-lab
    python kernels/exp_multistream.py

三个对照，看的不是"快了多少"，是"什么样的 kernel 才能重叠"：
    1. 4 个空 kernel：串行 vs 4 stream  → 应该几乎一样（启动是 CPU 单线程，重叠没意义）
    2. 4 个空转 kernel（各占 1 个 SM）：串行 vs 4 stream → 应该接近 4x（真并行）
    3. 4 个"打满带宽"的 kernel：串行 vs 4 stream → 应该几乎没收益（带宽已满）

第 3 个用你已有的 decode 太重，这里用 write1 的放大版不现实，就跳过——
记住判据即可：**并发收益 = 有多少资源空着 × 空着多久**。
"""

import time

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


def wall_us(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e6


def main():
    print(f"设备: {torch.cuda.get_device_name(0)}")
    mod = build()
    dev_stream = torch.cuda.current_stream()

    # ---------- 1. 空 kernel：串行 vs 4 流 ----------
    N = 2000
    t_serial = wall_us(lambda: [mod.empty_launch() for _ in range(N)])

    streams = [torch.cuda.Stream() for _ in range(4)]
    def four_stream_empty():
        for s in streams:
            with torch.cuda.stream(s):
                for _ in range(N // 4):
                    mod.empty_launch_on(s.cuda_stream)
    t_multi = wall_us(four_stream_empty)

    print(f"\n[1] {N} 个空 kernel")
    print(f"    1 条流串行 : {t_serial:9.1f} µs  → {t_serial/N:5.2f} µs/次")
    print(f"    4 条流并发 : {t_multi:9.1f} µs  → {t_multi/N:5.2f} µs/次")
    print(f"    → 结论：{'几乎没变（启动是 CPU 单线程，重叠无意义）' if abs(t_multi-t_serial)/t_serial < 0.2 else '有变化，想想为什么'}")

    # ---------- 2. 空转 kernel：串行 vs 4 流 ----------
    CYCLES = 2_650_000            # ≈1 ms @2.65 GHz
    t1 = wall_us(lambda: mod.spin_launch(CYCLES, dev_stream.cuda_stream))
    print(f"\n[2] 空转 kernel 单次 ≈ {t1:.0f} µs（目标 ~1 ms）")

    t_spin_serial = wall_us(lambda: [mod.spin_launch(CYCLES, dev_stream.cuda_stream) for _ in range(4)])

    def four_stream_spin():
        for s in streams:
            with torch.cuda.stream(s):
                mod.spin_launch(CYCLES, s.cuda_stream)
    t_spin_multi = wall_us(four_stream_spin)

    print(f"    4 个串行 : {t_spin_serial:9.1f} µs")
    print(f"    4 个并发 : {t_spin_multi:9.1f} µs")
    print(f"    → 加速比 {t_spin_serial/t_spin_multi:.2f}x（理想 4x；受 launch 开销和 SM 数限制）")

    print("""
────────────────────────────────────────────────
读法：
- 表 [1] 说明：kernel 太小（9 µs）时，多 stream 不能让总时间变短，
  因为瓶颈在 CPU 逐条提交，不在 GPU。
- 表 [2] 说明：kernel 够长（1 ms）且每块只占 1 个 SM 时，4 条流真的并行，
  加速比 ≈ 4。这是 stream 唯一的用处：**填满空闲资源**。
- 推论：你的 decode 是带宽型，DRAM 已 87.8%，4 条流并发只会更快打满，
  不会更快结束 → 所以 B 对你不适用（C 也只值 0.8%）。
────────────────────────────────────────────────
""")


if __name__ == "__main__":
    main()
