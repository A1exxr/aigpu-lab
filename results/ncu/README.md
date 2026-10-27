# Nsight Compute 剖面摘录

这里存的是每个 kernel 的 **ncu metrics 摘录**（不是原始 CSV，原始输出可用下面的命令复现）。
记录时间：2026-10-18 ~ 10-19，GPU：RTX 4060 Laptop（24 SM），`--clock-control none`。

## 为什么用 `--clock-control none`

WSL 下 ncu 默认会锁频（base clock）再跑，测出来的时间和真实运行不一致。
加 `--clock-control none` 才能看到接近实机的数字；代价是**跨次测量会有 ±5% 抖动**。

## 复现命令

```bash
export PATH=/usr/local/cuda-12.8/bin:$PATH        # ncu 2025.1.1 不在默认 PATH

# 单 kernel 全量指标（-c 1 只抓第一次命中，否则会抓 warmup 的几百次）
ncu --clock-control none --set full -c 1 \
    --kernel-name regex:fused_rmsnorm_residual_kernel \
    python kernels/bench_kernels.py --only fused --iters 1

# 只抓某几个指标（快）
ncu --clock-control none -c 1 \
    --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
launch__registers_per_thread,sm__warps_active.avg.pct_of_peak_sustained_active \
    python kernels/bench_kernels.py --only fused --iters 1
```

## 文件

| 文件 | kernel | 一句话结论 |
|---|---|---|
| [reduce.md](reduce.md) | `reduce_kernel` | DRAM 87.3% → 已贴墙，不能靠调参变快 |
| [softmax.md](softmax.md) | `softmax_kernel` | SM 88.0% / DRAM 38.2% → 卡在屏障，不是访存 |
| [fused_rmsnorm_residual.md](fused_rmsnorm_residual.md) | 融合版 + 两 kernel 版 | DRAM 92.8% → 到顶；融合省下 48 MiB 是实打实的 1.95× |

## 一个尚未解决的矛盾（诚实记录）

同一个融合版 kernel：

```
bench（CUDA event + L2 冲刷）  0.2826 ms  →  178.1 GB/s = 65.5% 峰值
ncu（--clock-control none）   199.90 µs  →  251.8 GB/s = 92.6% 峰值
```

**差 40%。** 已排除：降频（12 秒连跑 48060 次，`clocks.mem` 恒 8200 MHz、SM 2655 MHz、
77 W、60 °C）、CPU 入队（10.0 µs/次）、分配器开销（`empty_like` 0.2 µs）。
WSL 下单次 `cudaEvent` 计时会抖到 5–17 ms，所以 bench 走的是"多轮取中位数"。

**当前的处理方式**：看相对加速比用 bench（同一尺子），看瓶颈类型用 ncu（指标可信）。
两者绝对值不一致这件事本身记在 `docs/03-错题/01-CUDA错题本.md` 的 E-50。
