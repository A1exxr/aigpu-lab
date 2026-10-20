# ncu：fused_rmsnorm_residual（融合版 vs 两 kernel 版）

- 源文件：`kernels/w3_fused_rmsnorm_residual.cu`
- case：`(2048, 4096)` fp16，`out = rmsnorm(x + residual) * w`
- 命令：`ncu --clock-control none --set full -c 1 --kernel-name regex:rmsnorm python kernels/bench_kernels.py --only rmsnorm --iters 1`
- 记录：2026-10-18
- 口径：融合版真实流量 = **3N**（读 x、读 res、写 out）= 48 MiB；
  `w` 只有 8 KB 且常驻 L2，不计入。

## 融合版（1 个 kernel）

| 指标 | 值 | 读法 |
|---|---|---|
| Duration | **199.90 µs** | |
| **DRAM Throughput** | **92.83%** | 几乎打满，没有余量 |
| Compute (SM) Throughput | 29.54% | 计算单元很闲 |
| 有效带宽（自算） | 48 MiB ÷ 199.9 µs = **251.8 GB/s**（92.6%） | |
| Achieved Occupancy | 95.61%（理论 100%） | 占用率不是问题 |
| Registers / thread | 40 | 不紧张 |
| Static Shared Memory | 36 B | 只有 `warp_sum[8]` + 一个 rstd 标量 |
| Grid / Block | 2048 × 256 | 一行一个 block |
| Waves per SM | 14.22 | 够多，尾效应小 |
| Warp Cycles Per Issued Inst | 50.23 | |
| **Long Scoreboard stall** | **41.4%** | 主要在等显存回来 —— memory-bound 的正常长相 |
| L1 Hit / L2 Hit | 35.41% / 34.25% | 数据太分散，缓存救不了 |
| Mem Pipes Busy | 32.6% | 片上管线不忙 |

## 两 kernel 版（对照）

| kernel | Duration | DRAM | Compute(SM) | 自算带宽 |
|---|---|---|---|---|
| `add_residual_kernel` | 242.69 µs | **91.41%** | 15.72% | 207.4 GB/s（76.2%） |
| `rmsnorm_kernel` | 147.30 µs | 65.12% | 37.60% | 227.8 GB/s（83.8%） |
| 合计 | **390.0 µs** | | | 215.1 GB/s（79.1%） |

## 结论

```
融合 / 两 kernel = 390.0 ÷ 199.9 = 1.95×
理论字节比       = 96 MiB ÷ 48 MiB = 2.0×
```

**两块测量的比值和理论字节比吻合 → 融合省下的 48 MiB 是真实的收益，不是噪声。**

融合版到 92.8% 之后，剩下的 7% 不值得再花时间：它的三股流量（读 x、读 res、写 out）
已经是这个算法的下限，任何进一步的"优化"都只能是减少计算或改变算法本身。

## 附：两处 36 B 共享内存是干什么的

- `__shared__ float warp_sum[8]`：block 有 256 线程 = 8 个 warp，
  每个 warp 用 5 轮 `__shfl_down_sync` 归约出局部的 `sum(x²)`，lane 0 写进 `warp_sum[warp_id]`。
- `__shared__ float rstd`：warp 0 把 8 个部分和加起来、算出 `rstd = rsqrt(sum/N + eps)`，
  写这一个标量，再让全 block 读 —— **一个标量广播，用 4 B 换掉一次 HBM 往返和 8 次重复计算**。

整行 4096 个 fp16（8 KB）**不放共享内存**：没有复用，放进去只会多一次写读和一道屏障。
