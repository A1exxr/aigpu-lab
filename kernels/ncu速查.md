# ncu 速查（WSL 里用 Nsight Compute 看性能）

> 建立于 2026-10-18，配 w3_fused_rmsnorm_residual.cu 使用。

## 0. 可执行文件在哪

```bash
/usr/local/cuda-12.8/bin/ncu          # ← 完整路径，已经装了
export PATH=/usr/local/cuda-12.8/bin:$PATH   # 想直接敲 ncu 就加这行（可写进 ~/.bashrc）
```

`ncu --version` 报 not found 只是因为不在 PATH 上，不是没装。

## 1. 三条常用命令

```bash
NCU=/usr/local/cuda-12.8/bin/ncu
PY=/home/alex/miniconda3/envs/triton/bin/python
cd /home/alex/projects/aigpu-lab
```

### ① 关键五段（日常用这个）

```bash
$NCU --section SpeedOfLight --section MemoryWorkloadAnalysis \
     --section Occupancy --section LaunchStats --section WarpStateStats \
     --kernel-name regex:rmsnorm_kernel --launch-count 1 \
     $PY kernels/bench_kernels.py --only rmsnorm_residual
```

### ② 全量（几十项指标 + 优化建议，慢好几倍）

```bash
$NCU --set full -k regex:rmsnorm_kernel -c 1 \
    $PY kernels/bench_kernels.py --only rmsnorm_residual
```

### ③ 只要硬数字（可以直接抄进表格）

```bash
$NCU --metrics dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum \
     -k regex:rmsnorm_kernel -c 1 \
     $PY kernels/bench_kernels.py --only rmsnorm_residual
```

### 常用开关

| 开关 | 作用 |
|---|---|
| `-k regex:名字` | 只抓名字匹配的 kernel（正则，`|` 分隔多个） |
| `-c 1` / `--launch-count 1` | 只抓第 1 次。**bench 会跑几百次，不加能卡几十分钟** |
| `--section X` | 只要某几段（快） |
| `--set full` | 全量（慢，带优化建议） |
| `--clock-control none` | 不锁频，测真实频率下的表现（默认锁 base clock） |
| `--target-processes all` | 连子进程一起抓（load_inline 会起 ninja） |

## 2. 每段该看什么

| 段 | 关键指标 | 回答什么问题 |
|---|---|---|
| GPU Speed Of Light | `DRAM Throughput %`、`Compute (SM) Throughput %`、`Duration` | **带宽打满没有**？瓶颈在内存还是计算 |
| Memory Workload Analysis | L1/L2 Hit Rate、Mem Pipes Busy、sectors/req | 访存效率：有没有浪费带宽 |
| Occupancy | Theoretical / Achieved Occupancy | 被寄存器或共享内存卡住了吗 |
| Launch Statistics | Registers Per Thread、Static Shared Memory、Grid、Waves Per SM | 启动配置合不合理；寄存器会不会溢出 |
| Warp State | Warp Cycles Per Issued Instruction、各 stall 原因 | **到底在等什么**（等内存 = long scoreboard） |

读法要点：

- `DRAM Throughput` 接近 90% 以上 → 带宽已饱和，别再从计算侧优化。
- `Compute (SM) Throughput` 很低（<40%）而 DRAM 很高 → 纯访存 kernel，正常。
- `Achieved Occupancy` 接近 `Theoretical` → 占用率不是问题，别去调 block size。
- `Long Scoreboard` 占比高 → 在等显存返回；要提升的是**同时在飞的请求数**（向量化、提高 ILP、加大每线程处理量），不是减少指令总数。
- `Registers Per Thread` 涨到 64 以上要警惕 spill（local memory 会让带宽看着虚高或虚低）。

## 3. 2026-10-18 实测记录（fp16, 2048×4096 = 8.39M 元素，RTX 4060 Laptop）

### 融合版 `fused_rmsnorm_residual_kernel`（3N = 48 MiB）

```
Memory/DRAM Throughput   92.83 %      Compute (SM) 29.54 %
Duration                199.90 µs     SM 1.89 GHz（--clock-control none）
Occupancy               理论 100 % / 实测 95.61 %
Registers Per Thread    40            Static Shared Memory 36 B
Warp Cycles Per Issued  50.23         Long Scoreboard 占 41.4 %  ← 唯一瓶颈
L1 Hit 35.41 % / L2 Hit 34.25 % / Mem Pipes Busy 32.6 %
```

### 分步版两个 kernel（6N = 96 MiB 合计）

```
add_residual_kernel    Duration 242.69 µs   DRAM 91.41 %   Compute 15.72 %   (3N)
rmsnorm_kernel         Duration 147.30 µs   DRAM 65.12 %   Compute 37.60 %   (2N)
```

### 两把尺子的差别（重要）

| 实现 | bench 循环计时 | ncu 单次 Duration |
|---|---|---|
| 融合版 | 0.2826 ms（65.5%） | 199.9 µs（92.8%） |
| 分步版 | 0.3756 ms（98.5%） | 390.0 µs（242.7 + 147.3） |

- **ncu 尺子下比值 = 390.0 / 199.9 = 1.95×**，和字节比 6N/3N = 2.0 几乎一致。
- **bench 尺子下比值 = 1.33×**，因为 bench 里融合版被抬高了 40%，而分步版几乎没变。
- 已排除降频：12 秒连跑 48060 次，`clocks.mem` 恒 8200 MHz、SM 2655 MHz、77 W、60 ℃。
- 结论：**比较实现快慢只用一把尺子**。bench 用来看趋势（同一 harness 前后对比），ncu 用来回答“为什么”。

## 4. 坑

1. `-c 1` 不加，bench 的几百次 launch 会被逐个重放，极慢。
2. **`nsys` 在 WSL 下抓不到 GPU kernel**（只有 CPU 侧 CUDA API），报告里会出现 `does not contain CUDA kernel data`；这不是命令写错。要看 GPU 时间用 `ncu`。
3. `nsys` 的报告默认写 `/tmp`（WSL 内部），Windows 资源管理器走 `\\wsl.localhost\Ubuntu\tmp`，或 `cp` 到 `/mnt/c/...`。
4. `--clock-control none` 与默认（锁 base clock）的 Duration 会差 1–2%，但**别把两者的百分比混着比**。
5. WSL 下每次 launch 的 CPU 入队约 10–15 µs，单次计时用 `cudaEvent` 会测到 5–17 ms 的抖动，**单次计时不可信**，要连跑取平均。

## 5. GPU 存储分级（"SRAM" 就是共享内存 + L1）

GPU 没有一块叫 "SRAM" 的独立存储，片上存储是分层的。常说的 SRAM = **共享内存 + L1**，它们物理上是同一块 SRAM，按需切分。

| 层 | 容量（AD107 / 4060 Laptop） | 延迟量级 | 谁在用 | 对应 ncu 指标 | 要不要管 |
|---|---|---|---|---|---|
| 寄存器堆 | 每 SM 256 KB | ~1 周期 | 每个线程私有 | `Registers Per Thread` | **管**：>64 警惕 spill |
| **共享内存 / L1**（= SRAM） | 每 SM 128 KB，共享最多 100 KB | ~20–40 周期 | 同一 block 的线程 | `Static Shared Memory`、`L1 Hit`、`Mem Pipes Busy` | **管**：只在需要时开 |
| L2 | 约 24 MB | ~200 周期 | 整个 GPU 共享 | `L2 Hit Rate` | 基本不管，硬件自动 |
| 显存 DRAM | 8 GB / 272 GB/s | ~400–600 周期 | 全局 | `DRAM Throughput` | **管**：就是带宽 |

读法：延迟每往下一层，贵一个数量级以上。所有优化的本质都是**把访问留在更上层**，或者**减少下层的字节数**。

## 6. 什么时候开共享内存

判据表：

| 数据的去向 | 放哪 | 理由 |
|---|---|---|
| 只有本线程用 | **寄存器** | 最快，零延迟，不占共享内存 |
| 同一 block 内**多个线程要复用** ≥2 次 | **共享内存** | 只从显存读一次，之后从片上反复读 |
| 线程之间要**交换**数据（归约、转置） | **共享内存 + 屏障** | block 内唯一的通信手段 |
| warp 内部（32 线程）交换 | **shuffle 优先** | `__shfl_down_sync` 不用共享内存、不用屏障 |
| 跨 block 用 | **只能靠显存**（L2 兜底） | 共享内存跨 block 不可见 |

三条判据，满足任意一条才开：

1. **复用**：同一份数据被同一 block 里 ≥2 个线程读 ≥2 次（矩阵 tile 是典型）。
2. **交换**：线程之间要传值（归约的部分和、转置）。
3. **省显存**：不缓存就会重复读显存，而这份数据小到放得下。

**反例（重要）**：把整行 8 KB 塞进共享内存**不会更快**。共享内存带宽也只有约 128 B/时钟/SM，还会多一次"显存→共享内存"的搬运；当每个线程只碰自己那段数据（没有复用）时，直接读进寄存器更好。

现状对照：

| kernel | 共享内存 | 用途 | 判断 |
|---|---|---|---|
| `fused_rmsnorm_residual_kernel` | `warp_sum[8]` + `rstd` ≈ 36 B | 8 个 warp 的部分和 → 线程 0 汇总 | ✅ 只做交换，正确 |
| `rmsnorm_kernel`（分步） | 同样 36 B | 同上 | ✅ |
| `softmax_kernel` | `x_max[32]` + `sum_data[32]` = 256 B | warp 级部分和汇总 | ✅ |

三个 kernel 都**没有**把整行驻留到共享内存，因为那一行每个线程只碰自己的元素、没有复用 —— 这是对的。

## 7. 片上缓存什么时候看

判据只有一条：

```
DRAM Throughput > 85%          → 不用看片上，瓶颈在显存，调片上没用
DRAM Throughput 40% ~ 70% 且慢 → 才看 L1 / L2 / 共享内存
```

真要查，抓这三个指标：

```bash
NCU=/usr/local/cuda-12.8/bin/ncu
$NCU --kernel-name regex:你的kernel --launch-count 1 \
     --metrics l1tex__throughput.avg.pct_of_peak_sustained_elapsed,\
lts__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum \
     $PY kernels/bench_kernels.py --only 你的算子
```

| 指标 | 高了说明什么 | 怎么改 |
|---|---|---|
| `l1tex__throughput` > 80% | L1 / 共享内存这条管线堵了 | 减少共享内存往返、合并访存、少用 shfl |
| `lts__throughput` > 80% | L2 堵了 | 提高 L1 命中（数据分块复用） |
| `bank conflicts` > 0 | 共享内存写法的 bank 冲突 | 加 padding（`[32][33]` 之类）或换访问顺序 |

**注意**：共享内存的 bank conflict 只在真的用了共享内存且频繁访问时才有意义；本项目三个 kernel 的共享内存都只用于 8~32 个部分和，冲突可以忽略。

## 8. 算术强度 AI 与 Roofline（第一次听就看这一节）

### 8.1 一句话

**AI（算术强度）= 每搬 1 字节的数据，平均做多少次浮点运算。**

```
AI = 总 FLOPs ÷ 总字节数          单位：FLOP/B
```

它是**两个总量的比值**，不是时间关系，也不是"同一个字节被算了几次"。

### 8.2 Roofline 模型：回答"卡在搬数据还是算运算"

Roofline（屋顶线）把两条上限画在同一张图上：

```
性能 (FLOPS)
  ↑
  │                          ┌───────────────────  ← 算力上限 65.2 TFLOPS
  │                        ／
  │                     ／     ← 斜线：带宽限制，斜率 = 带宽 272 GB/s
  │                  ／
  │               ／
  └──────────────┴──────────────────────────────→ AI (FLOP/B)
                 ↑
              拐点 239.7
```

- **横轴**：AI（每字节做多少次运算）
- **纵轴**：实际能达到的 FLOPS
- **斜线**：带宽限制区。AI 越低，数据供不上，性能被压在斜线上
- **水平线**：算力限制区。AI 再高，性能也卡在 65.2 TFLOPS
- **拐点**：两条线的交点

```
拐点 = 65.2e12 ÷ 272e9 = 239.7 FLOP/B
```

**你的 kernel 落在斜线上 → memory-bound；落在水平线上 → compute-bound。**

### 8.3 单位 T / G / M 是什么

| 单位 | 数量级 | 读法 |
|---|---|---|
| K | 10³ | 千 |
| M | 10⁶ | 百万 |
| G | 10⁹ | 十亿 |
| **T** | **10¹²** | **万亿** |
| P | 10¹⁵ | 千万亿 |

```
65.2 TFLOPS = 65.2 × 10¹² 次浮点运算/秒 = 每秒 65.2 万亿次
272 GB/s    = 272 × 10⁹ 字节/秒        = 每秒 2720 亿字节
```

### 8.4 "AI = 31" 到底是什么

意思是：**这个算法总共搬 N 字节，同时做 31N 次运算。**

不是"搬 1 字节的时间能做 31 次"，也不是"1 字节被算了 31 次"——是全局统计。

用 naive attention 算一遍（**B=1, H=1, S=1024, D=64, fp16**）：

| 操作 | 字节 |
|---|---|
| 读 Q | 2SD |
| 读 K | 2SD |
| 读 V | 2SD |
| 写 S_mat（QKᵀ 结果） | 2S² |
| 读 S_mat（softmax 输入） | 2S² |
| 写 P（softmax 结果） | 2S² |
| 读 P（P·V 输入） | 2S² |
| 写 O | 2SD |
| **合计** | **2(4SD + 4S²)** |

> 注：V 只读一次。若把"读 V"记两次会得到 2(5SD+4S²)，结果差 ~0.3，不影响结论。

```
字节 = 2(4×1024×64 + 4×1024²) = 8.91e6 B ≈ 8.5 MiB
FLOPs = 2S²D（QKᵀ）+ 2S²D（PV） = 4S²D = 2.68e8
AI = 2.68e8 ÷ 8.91e6 ≈ 30
```

**AI 只由算法结构决定，与代码写得好坏、与硬件都无关。** 所以：
同一份 attention，naive 把 S² 写回显存 → 字节巨大 → AI 低；
FlashAttention 把 S² 留在片上 → 字节骤减、FLOPs 不变 → AI 高。

### 8.5 每秒视角：为什么"算力闲着"

带宽每秒最多送 272 GB 进来。每字节配套 30 次运算，那么这些数据最多只够做：

```
30 × 272e9 = 8.2e12 = 8.2 TFLOPS
算力上限 65.2 TFLOPS
利用率 = 8.2 / 65.2 ≈ 13%
```

**87% 的算力在等数据——这就是 memory-bound 的量化含义。**

反过来 FlashAttention：`AI ≈ 512 → 512 × 272e9 = 139 TFLOPS`，超过 65.2 → 数据供得比算得快 → **compute-bound，算力才是瓶颈**。

### 8.6 两条可背的公式

```
naive attention:  AI = (D/2) × S/(S+D)   →  S ≫ D 时趋近 D/2 = 32
FlashAttention:   AI = S/2               →  S 越大 AI 越高（S=1024 → 512）
```

这解释了为什么**序列越长，FlashAttention 的优势越大**：naive 的 AI 封顶在 32，flash 的 AI 随 S 线性增长。

### 8.7 两个数怎么拿（实测口径）

- **FLOPs**：优先按算法数（一次乘加算 2 次）。实测用 ncu：
  ```
  sm__sass_thread_inst_executed_op_ffma_pred_on.sum × 2
  + sm__sass_thread_inst_executed_op_fmul_pred_on.sum
  + sm__sass_thread_inst_executed_op_fadd_pred_on.sum
  ```
- **字节**：`dram__bytes_read.sum + dram__bytes_write.sum`

### 8.8 我已经算过的四个（备忘）

| kernel | 字节 | FLOPs | AI | 判定 |
|---|---|---|---|---|
| `fused_rmsnorm_residual` (2048×4096) | 48 MiB | 3.4e7 | **≈0.68** | 极度 memory-bound |
| `rmsnorm_kernel` (同上) | 32 MiB | 1.7e7 | **≈0.5** | 同上 |
| `attn_naive` (S=2048) | 1056 MiB | 3.44e10 | **≈31** | 仍 memory-bound |
| FlashAttention (S=2048) | 32 MiB | 3.44e10 | **≈1024** | compute-bound |

**代码优化（float4、合并访存、调 block size）只影响"实际带宽能多接近 272 GB/s"，不改变 AI。**
要改 AI，只能改算法：融合、分块、复用。

### 8.9 一句话总结

| 问题 | 答案 |
|---|---|
| Roofline 是什么 | 用 AI 判断 kernel 卡在带宽还是算力的模型 |
| T 是什么 | Tera，10¹²，万亿 |
| AI = 31 是什么 | 总 FLOPs ÷ 总字节 = 31，每搬 1 字节对应 31 次运算 |
| 是时间关系吗 | ❌ 不是，是全局比值 |
| 每秒视角 | 272 GB/s × AI 31 ≈ 8.2 TFLOPS，远低于 65.2，算力闲着 |
| AI 只看算法吗 | 是。代码只影响实际带宽能多接近峰值 |
| 减字节有什么用 | 直接抬高 AI，把瓶颈从带宽转到算力（融合就是这么起效的） |

## 9. 访存粒度：cache line（128 B）vs sector（32 B）

### 9.1 两个概念不是一回事

| 概念 | 大小 | 作用 |
|---|---|---|
| **cache line** | 128 B | **管理 / 查表单位**：L1 里每个槽位记一条 line 的地址高位（tag）。查表、命中判断按 line 走 |
| **sector** | 32 B | **搬运单位**：L2 ↔ L1、DRAM ↔ L2 真正搬的最小粒度。**1 line = 4 sector** |

为什么还要 line：tag 按 line 存，如果按 sector 存，tag 数量 ×4，硬件成本爆炸。所以「用 line 查表，用 sector 搬数据」。

### 9.2 sectored cache：只搬被碰到的那个 sector

现代 NVIDIA GPU 是 **sectored** 的：一条 128 B line 里，哪个 32 B sector 被访问了，才把那个 sector 拉上来，其余 3 个原地不动。

```
一条 line：
┌─────────┬─────────┬─────────┬─────────┐
│sector 0 │sector 1 │sector 2 │sector 3 │  共 128 B
│  32 B   │  32 B   │  32 B   │  32 B   │
└─────────┴─────────┴─────────┴─────────┘
      ↑
   只碰了这 2 B → 只有 sector 0 被搬
```

类比：line 是一排 4 格快递柜，sector 是其中一格。你只买了 1 格里的东西，快递员只开那一格，不会把整排 4 格都搬来。

### 9.3 两种口径算浪费率（w4 `qk_kernel` 旧版为例）

访问方式：32 个线程各读 2 B（fp16），地址彼此间隔 128 B → 落在 32 条不同 line 上。

| 口径 | 算式 | 结果 |
|---|---|---|
| A. 按 **line** 算 | 32 条 line × 128 B = 4096 B；有用 32 × 2 = 64 B | **1.56%** |
| B. 按 **sector** 算 | 32 条 line × 1 个 sector × 32 B = 1024 B；有用 64 B | **6.25%** |

- **DRAM 带宽按 sector 计** → 真实浪费率是 **6.25%**，不是 1.56%。
- 1.56% 只在「非 sectored cache / 整 line 预取 / 写-分配」这几种情况下才对，现代 GPU 读路径不是这样。
- 面试说法：「按 32 B sector 粒度算，有效字节只有 6.25%」——点出粒度会加分。

### 9.4 通用判据：sectors / request

```
理想 sectors/request = 每线程访问的字节数（因为 32 线程 × N 字节 / 32 B = N）
```

| 每线程访问 | 合并后的 warp 请求 | sectors/request | 有效字节 |
|---|---|---|---|
| fp16 × 1 个（2 B） | 64 B 连续 | **2** | 100% |
| fp32 × 1 个（4 B） | 128 B 连续 | **4** | 100% |
| fp16 × 2 个（half2，4 B） | 128 B 连续 | **4** | 100% |
| w4 旧版 qk（跨 32 条 line） | 32 条不同 line | **32** | 6.25% |

**结论：sectors/request 越接近「每线程字节数」越好；远超它说明访存没合并。**

### 9.5 在 ncu 里量

```bash
ncu --metrics \
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum \
python your_script.py
```

- `requests` = warp 一共发出几次 load
- `sectors` = 实际搬了多少个 32 B
- 两者相除就是上表的 sectors/request
- 还有 `l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum`（写）、`lts__t_sectors_op_read.sum`（L2 读，看 sector 有没有被重复搬）

### 9.6 三句话

1. **line = 128 B，管查表；sector = 32 B，管搬数据。**
2. 浪费率按 sector 算：w4 旧版 qk 是 **6.25%**，不是 1.56%。
3. 判合并好不好看 **sectors/request**，理想值 = 每线程字节数（fp16 单元素 = 2，fp32 或 half2 = 4）。

## 10. ILP（指令级并行）：为什么「一次多算几个」反而更快

### 10.1 一句话

**ILP = 让同一个线程里有多条互不依赖的指令同时在飞。** 处理器不必等上一条算完，就能发下一条。

### 10.2 反例：串行依赖链（qk 的第一版）

```cuda
acc += __shfl_down_sync(0xffffffff, acc, 16);   // 用 acc，产出 acc
acc += __shfl_down_sync(0xffffffff, acc, 8);    // 必须等上一轮
acc += __shfl_down_sync(0xffffffff, acc, 4);
acc += __shfl_down_sync(0xffffffff, acc, 2);
acc += __shfl_down_sync(0xffffffff, acc, 1);
```

5 轮全是「必须排队」的操作，每轮延迟 20–30 周期 → 一个 warp 每算一个 j 就要干等 **100–150 周期**，而它只有约 20 条指令可发 → 发射效率约 **16%**。

### 10.3 正例：4 条互不依赖的链

```cuda
float a0, a1, a2, a3;              // 4 个不同 j 的部分和，互不依赖
a0 += __shfl_down_sync(0xffffffff, a0, 16);
a1 += __shfl_down_sync(0xffffffff, a1, 16);   // ← 不用等 a0，立刻能发
a2 += __shfl_down_sync(0xffffffff, a2, 16);
a3 += __shfl_down_sync(0xffffffff, a3, 16);
```

a0 在等结果时，调度器去发 a1/a2/a3 → 原本的空洞被填上。

### 10.4 延迟 vs 吞吐（这句要背）

- **延迟（latency）**：一条指令从发出到结果可用要多久。shuffle ≈ 20–30 周期；HBM ≈ 400–600 周期
- **吞吐（throughput）**：单位时间能发多少条
- 只有 1 条链 → 只能「等延迟」，吞吐上不去
- 多条独立链 → 用「别人在等」的时间发别人的指令 → **把延迟隐藏成吞吐**

### 10.5 发射效率怎么算

```
发射效率 = 理论最短时间 ÷ 实测时间
理论最短时间 = warp 指令总数 ÷ (SM 数 × 每 SM 调度器数 × 主频)
```

RTX 4060 Laptop：24 SM × 4 调度器 × 2.655 GHz ≈ **255G 条 warp 指令/秒**

qk 实测：670M 条 ÷ 255G ≈ 2.6 ms 理论，实测 16 ms → **效率 16%**

### 10.6 ILP 的代价

- 每条「在飞的链」要占一个累加器寄存器 → 4 条链 = 4 个累加器 + 4 份地址计算
- 寄存器涨 → occupancy 可能掉 → 又和 §11 打架
- 经验：ILP 2–4 最划算，8 以上一般开始掉

---

## 11. 寄存器、Occupancy、MIO：三个总被问到的概念

### 11.1 寄存器多少算多？

不看绝对值，看**能不能装满 SM**：

```
每 SM 寄存器总数（Ada）= 65536 个（32-bit）
每 SM 最大常驻线程     = 1536
⇒ 想 100% occupancy：每线程 ≤ 65536 ÷ 1536 ≈ 42 个
```

| 每线程寄存器 | 256 线程/block 能装几个 block | occupancy |
|---|---|---|
| 32 | 8 | 100% |
| 40 | 6 | 100%（6×256 = 1536） |
| **72** | **3** | **50%** |
| 96 | 2 | 33% |

所以「72 个算多吗」→ 单看不算多（硬件上限 255），但**对 100% occupancy 来说太多**。

### 11.2 怎么降寄存器（按有效度排）

1. 减少同时活着的变量（用完就让它死）
2. 减小 `#pragma unroll` 的展开倍数
3. 拆 kernel（一个 kernel 别干三件事）
4. `__launch_bounds__(256, 6)`：让编译器自己压到能装 6 个 block

❌ **无效做法**：改 block 数 / 改 grid — 寄存器是 **per-thread** 的，跟 block 数量无关。

### 11.3 MIO Throttle 是什么

MIO = Memory Input/Output 队列。**共享内存、shuffle、特殊函数（exp/sin）、部分 L1 访问**都排这条队。队列满了 → 指令发不进去 → 记为 `stall_mio_throttle`。

对应动作：
- 查共享内存 bank conflict
- 小规模归约改用 shuffle 而不是共享内存
- 减少共享内存读写次数（一次读 4 个，而不是读 4 次）

### 11.4 Short / Long Scoreboard 怎么区分

| 指标 | 在等什么 | 说明 |
|---|---|---|
| `short_scoreboard` | **片上**（shared memory / L1 / MIO） | 卡在片上 |
| `long_scoreboard` | **显存**（HBM，或 L2 miss 之后） | 卡在显存 |

这是区分「片上瓶颈」和「显存瓶颈」最快的两个指标。

### 11.5 「L2 挡不挡得住」是什么意思

一次 kernel 发出的读取请求，如果数据还在 L2 里 → **命中，不打 DRAM**；如果已被别的数据挤出去 → **miss，必须回 DRAM**。

所以「挡不挡得住」= **这份数据的重复读取能不能被 L2 吸收**。

qk 的例子：每个 block 读 128 KB 的 `K[b,h]`，共 32768 个 block → 4 GiB 请求；但 `K[b,h]` 只有 128 KB，24 MiB 的 L2 能装 **192** 个（`24×1024÷128 = 192`），所以只要同一 `(b,h)` 的 block 连续执行，就只有第一次 miss。

保证命中的条件：**grid 排布让 `(b,h)` 相同的 block 在时间上连续**——`row = ((b*H + h)*S + i)` 正好满足。

## 12. 一页总结（记这 9 句）

1. 看带宽 = 看 `DRAM Throughput`；和峰值 272 GB/s 比。
2. 片上 = 寄存器 / 共享内存+L1（SRAM）/ L2；只在 DRAM 没喂满时才查。
3. 共享内存只为"复用"和"交换"而开，不为"缓存"而开。
4. warp 内部交换优先用 shuffle，比共享内存快且省屏障。
5. 先确认尺子（bench 循环 vs ncu 单次），再谈优化。
6. 瓶颈判断的根子是 **AI = FLOPs ÷ 字节**：<239.7 是带宽问题，>239.7 是算力问题。
7. 浪费率按 **sector（32 B）**算，不是 line（128 B）；合并好坏看 sectors/request 是否等于每线程字节数。
8. **一次多算几个（ILP）** 是用独立链填 shuffle/访存的延迟洞；发射效率 = 理论时间 ÷ 实测时间。
9. 寄存器每线程 ≤42 才可能 100% occupancy；`mio_throttle` = 片上队列满，`short_scoreboard` = 等片上，`long_scoreboard` = 等显存。
