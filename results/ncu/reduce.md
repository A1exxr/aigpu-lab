# ncu：reduce_kernel

- 源文件：`kernels/w2_reduce.cu`
- case：`(4194304,)` fp16，1D 全归约到标量
- 命令：`ncu --clock-control none --set full -c 1 --kernel-name regex:reduce_kernel python kernels/bench_kernels.py --only reduce --iters 1`
- 记录：2026-10-18

## 关键指标

| 指标 | 值 | 读法 |
|---|---|---|
| Duration | **84.03 µs** | 单次实际耗时 |
| **DRAM Throughput** | **87.25%** | 显存带宽已用掉 87% → 贴着墙 |
| Compute (SM) Throughput | 39.08% | 计算单元只忙 39%，不是算力瓶颈 |
| `dram__bytes_read.sum` | 19.20 MB | 比理论 8 MB 多（warmup/L2 冲刷残留） |
| Registers / thread | 16 | 很低，没有寄存器压力 |
| Warps Active | 94.88% | 占用率没问题 |
| Warp Cycles Per Issued Inst | — | 没有明显 stall（等显存是主要来源） |

## 判定

**DRAM-bound，且已贴墙（87%）。** 结论是：这个 kernel **不能再靠调参变快**，
唯一的优化方向是减少搬运的字节（fp16 存 fp16 已经是 2N=16 MB 的下限）。

`Compute 39%` 说明 SM 有一半时间在等数据——这正是 memory-bound 的典型特征：
**不要看到 39% 就以为"计算没吃满，可以加点活干"**，加活只会更慢。

## 相应的小结

- 归约用两级：warp 内 `__shfl_down_sync`（5 轮）→ 每 warp 一个部分和写共享内存 → warp 0 再归约。
- 相比"共享内存树形归约"，屏障从 `log2(blockDim)` 次降到 1 次。
- bench 报 54.3%（147.6 GB/s）与 ncu 的 87.25% 是两把尺子，见 [README.md](README.md) 的说明。
