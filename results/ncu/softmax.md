# ncu：softmax_kernel

- 源文件：`kernels/w2_softmax.cu`
- case：`(1, 2048, 4096)` fp16，按最后一维做 softmax
- 命令：`ncu --clock-control none --set full -c 1 --kernel-name regex:softmax_kernel python kernels/bench_kernels.py --only softmax --iters 1`
- 记录：2026-10-18

## 关键指标

| 指标 | 值 | 读法 |
|---|---|---|
| Duration | 0.216 ms（bench 中位数） |  |
| **Compute (SM) Throughput** | **87.96%** | 计算/发射管线接近饱和 |
| **DRAM Throughput** | **38.20%** | 显存只用了 38%，远没喂饱 |
| 有效带宽（bench 口径） | 155.4 GB/s = 57.1% | 算法最小流量 = 2N |

## 判定

**SM-bound，不是 memory-bound。** SM 88% 而 DRAM 38%，说明瓶颈在指令发射与同步，
不在显存。继续优化访存不会有用。

这也解释了为什么"减屏障"是唯一有效的优化：

| 版本 | 屏障数 | 延迟 | 带宽 |
|---|---|---|---|
| 共享内存树形归约 | 20 | 0.2478 ms | 49.8% |
| warp 内 shuffle（两级） | 10 | 0.2212 ms | 55.8% |
| 合并屏障（最终版） | **4** | **0.2160 ms** | **57.1%** |

每去掉一次屏障：级别 1 省 2.66 µs，级别 2 只省 0.87 µs —— **收益递减**，
这也是停手（不再追求 2 次屏障）的依据。

## 教训

`DRAM 38%` 出现时，第一反应不该是"换个访存写法"，而是先问：
**这台机器上还有什么东西饱和了？** 这里答案是 `sm__throughput 88%`（屏障 + 指令）。
