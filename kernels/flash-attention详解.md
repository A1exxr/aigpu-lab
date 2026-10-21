# FlashAttention 详解：从显存流量到 online softmax

> 写给"已经能写 naive attention、但说不清 FlashAttention 为什么快"的人。
> 本文所有数字都用本机配置算：**RTX 4060 Laptop / 272 GB/s / 65.2 TFLOPS(fp16) / Roofline 拐点 239.7**，
> 张量形状 `B=4, H=8, S=1024 或 2048, D=64, fp16`——也就是 `kernels/w4_attn_naive.cu` 用的那组。

---

## 0. 一句话与三句补充

**一句话：FlashAttention 是把 attention 的 `S×S` 中间矩阵按块算、算完即扔，从而把 HBM 流量从 O(S²) 降到 O(S²d²/M) 的"IO 感知"算法。**

三句必须同时说清，否则就是背概念：

1. **它是精确算法，不是近似。** 数学上与标准 attention 逐元素等价（fp16 下误差 ~1e-3）。
2. **它不减少矩阵乘的 FLOPs。** 前向的 matmul 次数和标准实现一样，甚至因为重算略增。
3. **它快，是因为少搬了数据**：把"写 S² → 读 S² → 写 P → 读 P"这四次 HBM 往返，换成片上 SRAM 里的分块计算。

---

## 1. 先算清 naive 的三笔账

### 1.1 标准 attention 的三步

```
S = Q · Kᵀ · scale        (B,H,S,S)      中间矩阵，必须落 HBM
P = softmax(S, dim=-1)    (B,H,S,S)      又一个，必须落 HBM
O = P · V                 (B,H,S,D)
```

`S` 和 `P` 是 `S×S`——**它们的元素个数与序列长度的平方成正比**，这就是所有问题的根源。

### 1.2 流量账（本机配置，fp16）

单个 `(b,h)` 的 `S×S` 矩阵：`1024×1024×2 B = 2 MiB`；`B×H = 32` 个 → **64 MiB**。

naive 三 kernel 各读写一次：

| 动作 | S=1024 | S=2048 |
|---|---|---|
| 写 S（QKᵀ 输出） | 64 MiB | 256 MiB |
| 读 S（softmax 输入） | 64 MiB | 256 MiB |
| 写 P（softmax 输出） | 64 MiB | 256 MiB |
| 读 P（PV 输入） | 64 MiB | 256 MiB |
| Q/K/V 读 + O 写 | 16 MiB | 32 MiB |
| **合计** | **272 MiB** | **1056 MiB** |

### 1.3 时间账 vs 算力账

```
S=1024：带宽下限 272 MiB ÷ 272 GB/s = 1.05 ms      算力 8.59 GFLOP ÷ 65.2 TFLOPS = 0.13 ms
S=2048：带宽下限 1056 MiB ÷ 272 GB/s = 4.07 ms     算力 34.36 GFLOP ÷ 65.2 = 0.53 ms
```

**带宽下限是算力下限的 8 倍。** 算术强度 `AI = FLOPs ÷ 字节 ≈ 30`，拐点是 **239.7**——
**差 8 倍，所以 naive attention 在这张卡上必然是 memory-bound，与代码写得好坏无关，只与"搬多少字节"有关。**

> 这也是 `w4_attn_naive.cu` 的实测结论：15.53 ms 离 1.05 ms 还差 15 倍，但那 15 倍不是"没喂满带宽"，
> 而是**指令发射和 L1 扇区**的问题（AI 极低时，连"把数据搬到 L1"这件事本身都成了瓶颈）。

---

## 2. 核心思想：两件事拼起来

### 2.1 分块（tiling）

把 Q 按行切成 `BLOCK_M` 行一块，K/V 按行切成 `BLOCK_N` 行一块：

```
对每个 Q 块 q（BLOCK_M × D）：
    初始化 m、l、acc
    for 每个 K/V 块 k, v：
        s = q · kᵀ            (BLOCK_M × BLOCK_N)   ← 只存在于片上
        用 s 更新 m、l、acc
    O 块 = acc / l
```

`S×S` 的大矩阵**从来没有被完整构造出来**，每一块 `s` 用完就丢。

### 2.2 障碍：softmax 需要整行的 max 和 sum

softmax 的定义要求知道整行的 `max` 和 `Σexp`，可分块时行还没算完。

**解法：online softmax（在线归一化）——边算边更新，随时保持"如果现在结束，结果是对的"。**

这两件事拼起来，才是 FlashAttention。**只做分块不做 online softmax，结果就是错的。**

---

## 3. online softmax 完整推导

### 3.1 目标

对一行分数 `x = [x_1...x_S]`，输出

```
O = Σ_i exp(x_i - m) · v_i  /  Σ_i exp(x_i - m)        m = max(x)   （数值稳定性用）
```

**关键观察：`m` 取任何值结果都一样**（分子分母同乘 `exp(m' - m)`，约掉）。它只影响数值范围，不影响数学结果。

这条"不变量"是 online 递推合法的根本原因。

### 3.2 三个状态

| 状态 | 定义 | 形状 |
|---|---|---|
| `m` | 到目前为止见过的最大分数 | `(BLOCK_M,)` |
| `l` | `Σ exp(x_j - m)`，已按当前 `m` 归一 | `(BLOCK_M,)` |
| `acc` | `Σ exp(x_j - m) · v_j` | `(BLOCK_M, D)` |

### 3.3 递推（处理新块 `s`，形状 `BLOCK_M × BLOCK_N`）

```
m_new = max(m_old, rowmax(s))
alpha = exp(m_old - m_new)                     ← 旧结果的"重新归一化因子"
l_new = l_old · alpha + rowsum(exp(s - m_new))
acc_new = acc_old · alpha + exp(s - m_new) · v
m = m_new
```

**`alpha` 是全部的关键。** 因为 `l_old` 和 `acc_old` 是按 `exp(x - m_old)` 累积的，现在分母基准换成了 `m_new`，
旧结果必须乘 `exp(m_old - m_new)` 才能与新结果同口径。

注意 `alpha ≤ 1`（因为 `m_new ≥ m_old`），所以只会"缩小"，不会溢出。

### 3.4 数值例子（两块，纯手算）

分数 `x = [2, 0, 1, -1]`（第 1 块 `[2,0]`，第 2 块 `[1,-1]`），value `v = [10, 20, 30, 40]`。

**整体做法：**

```
m = 2
w = [e^0, e^-2, e^-1, e^-3] = [1.000, 0.1353, 0.3679, 0.0498]
l = 1.5530
acc = 1.000×10 + 0.1353×20 + 0.3679×30 + 0.0498×40 = 10 + 2.706 + 11.036 + 1.992 = 25.734
O = 25.734 / 1.5530 = 16.570
```

**分块做法：**

```
块 1：s = [2, 0]
  m     = 2
  l     = e^0 + e^-2 = 1.1353
  acc   = 1.000×10 + 0.1353×20 = 12.706

块 2：s = [1, -1]
  m_new = max(2, 1) = 2                       ← 没变大
  alpha = exp(2 - 2) = 1.0                    ← 旧结果无需缩放
  l     = 1.1353×1.0 + (e^(1-2) + e^(-1-2))   = 1.1353 + (0.3679 + 0.0498) = 1.5530
  acc   = 12.706×1.0 + (0.3679×30 + 0.0498×40) = 12.706 + 13.028 = 25.734
  O     = 25.734 / 1.5530 = 16.570            ← 与整体做法完全一致
```

**再试一个 `m` 会变大的例子**（这是 `alpha` 真正起作用的地方）：块 1 `[0, -1]`，块 2 `[2, 1]`。

```
块 1：m = 0, l = 1 + 0.3679 = 1.3679, acc = 10 + 0.3679×20 = 17.358
块 2：m_new = 2, alpha = exp(0-2) = 0.1353
      l   = 1.3679×0.1353 + (e^0 + e^-1) = 0.1851 + 1.3679 = 1.5530
      acc = 17.358×0.1353 + (1×30 + 0.3679×40) = 2.3485 + 44.716 = 47.065
      O   = 47.065 / 1.5530 = 30.31
```

整体验算：`w=[e^-2, e^-3, e^0, e^-1]=[0.1353,0.0498,1,0.3679]`，`l=1.5530`，
`acc = 0.1353×10+0.0498×20+1×30+0.3679×40 = 1.353+0.996+30+14.716 = 47.065` → `O=30.31` ✅

**两个例子都相等——这不是巧合，是因为 `alpha` 精确地补偿了基准的变换。**

### 3.5 正确性的一句话证明

`l` 是 `Σ exp(x_i - m)` 的**可结合**累积（求和可任意分段），`m` 的切换只改变公共因子
`exp(m' - m)`，而分子分母同时缩放不改变比值。**所以分块结果与整体结果恒等。**

### 3.6 为什么要减 `m`（数值稳定性）

fp16 的 `exp` 在 `x > 11.09` 时溢出（上限 65504），fp32 在 `x > 88` 溢出。
attention 分数 = `Q·Kᵀ/√D`，D=64 时点积范围可达 ±100。**不减 `max` 必然溢出成 inf。**
online 版本里 `m` 是滚动最大值，`exp(x - m) ≤ 1` 恒成立——**天然不溢出**。

---

## 4. 伪代码（论文 Algorithm 1，逐行注释）

```
输入：Q, K, V ∈ R^{S×d}（单个 head），块大小 B_m, B_n
输出：O ∈ R^{S×d}，L = 每行的 logsumexp

把 Q 切成 T_m = ⌈S/B_m⌉ 块，K/V 切成 T_n = ⌈S/B_n⌉ 块
并行 for 1 ≤ i ≤ T_m：                      ← grid 维 1（每个 Q 块一个程序）
    q_i = Q 的第 i 块                       ← 读 B_m×d
    O_i = 0, l_i = 0, m_i = -∞              ← 片上初始化
    for 1 ≤ j ≤ T_n：                       ← 串行循环（这是"循环"不是并行）
        k_j, v_j = K/V 的第 j 块            ← 读 B_n×d（每个 i 都要重读一遍！）
        s_ij = q_i · k_jᵀ                   ← (B_m, B_n)，只存在于寄存器/共享内存
        m_ij = rowmax(s_ij)
        m_new = max(m_i, m_ij)
        p_ij = exp(s_ij - m_new)            ← (B_m, B_n)
        alpha = exp(m_i - m_new)            ← (B_m,)
        l_i = l_i · alpha + rowsum(p_ij)
        O_i = O_i · alpha + p_ij · v_j      ← (B_m, d)
        m_i = m_new
    O_i = O_i / l_i                         ← 最后一次性除
    写回 O 的第 i 块
```

**读一遍这份伪代码，数一下 HBM 访问：**
- Q 读一次、O 写一次（各 `S·d`）
- K、V **每个 Q 块都要读一遍** → 共 `T_m · 2 · S · d` 次

### 4.1 所以"K/V 只读一次"是常见的错误说法

**不是。K/V 被读了 `T_m = S/B_m` 次。**

真实流量（单个 head）：

```
Q + O          2·S·d
K + V          2·S·d·(S/B_m)
合计           ≈ 2·S²·d/B_m
```

标准实现是 `Θ(S²)`（还要乘 4，因为 S、P 各写一次读一次）。
**FlashAttention 的 `S²` 项被除以了 `B_m`**——这就是 `O(S²d²/M)` 这一族结论的来源（`M ≈ B_m·B_n·d` 级别的片上容量）。

---

## 5. 流量对比：把数算到底

配置 `B=4, H=8, D=64, fp16`，单个 `(b,h)` 的 Q/K/V/O 各 `S·64·2 B`。

### 5.1 S=1024

| 方案 | 流量构成（单个 b,h） | 流量 | ×32 个 (b,h) |
|---|---|---|---|
| naive（4 次 S² 往返） | 4×2 MiB + 0.5 MiB | 8.5 MiB | **272 MiB** |
| Flash `B_m=128` | K/V 读 8 次 = 8×256 KiB = 2 MiB，Q 读 + O 写 0.25 | 2.25 MiB | **72 MiB** |
| Flash `B_m=64` | K/V 读 16 次 = 4 MiB + 0.25 | 4.25 MiB | **136 MiB** |

**S=1024 时，Flash 的收益只有 2× ~ 3.8×，不是几十倍。** 因为 S 小的时候 `S²` 项本来就不占绝对主导。

### 5.2 S=2048

| 方案 | 流量 | 倍数 |
|---|---|---|
| naive | 1056 MiB | 1× |
| Flash `B_m=128` | 16 次 × 2 × 256 KiB = 8 MiB + 0.5 → 8.5 MiB ×32 = **272 MiB** | **3.9×** |
| 理想（`S²` 项归零的下界） | 32 MiB | 33× |

**结论：S 越大，Flash 的优势越大（比值 ≈ `4·S·B_m / (2S + 2B_m)` 量级），
但"33×"是 `S²` 项完全消失的极限，实际做不到——因为 K/V 必须被重读。**

> 这一段是简历和面试里最容易说错的地方。**说"少搬了 4 次 S² 往返"是对的；
> 说"K/V 只读一次、总流量降到 1/30"是错的。** 真实收益取决于 `S` 和 `B_m`。

### 5.3 算术强度怎么变

`FLOPs = 4·B·H·S²·D`（两个 matmul，各 `2S²D`），S=1024 时 `8.59 GFLOP`。

| 方案 | 流量 | AI | 判定 |
|---|---|---|---|
| naive | 272 MiB | **30** | memory-bound（拐点 239.7，差 8 倍） |
| Flash `B_m=128` | 72 MiB | **114** | 仍 memory-bound，但已接近拐点 |
| Flash `B_m=256` | ~40 MiB | **205** | 接近拐点 |
| S=2048 的 Flash | 272 MiB，FLOPs 34.36 G | **120** | 同上 |

**AI 从 30 提到 114，就是"同样算力下少搬 3 倍数据"的直接结果。**

---

## 6. 片上存储预算：为什么 2022 年才做得到

`BLOCK_M=64, BLOCK_N=64, D=64, fp16`：

| 数据 | 大小 |
|---|---|
| q 块 `64×64 fp16` | 8 KiB |
| k 块 `64×64 fp16` | 8 KiB |
| v 块 `64×64 fp16` | 8 KiB |
| `acc` `64×64 fp32` | 16 KiB |
| `m, l` 各 64 个 fp32 | 0.5 KiB |
| **合计** | **≈ 40 KiB** |

AD107 每 SM：共享内存 128 KiB（单 block 上限 100 KiB）、寄存器 64K。
**40 KiB 装得下 → 一个 SM 可以同时跑 2~3 个这样的 block。**

`BLOCK_M=128` 时 `acc` 变 32 KiB、q 变 16 KiB，合计约 72 KiB——仍可，但只剩 1 个 block。

**这就是"分块大小"的本质权衡：块越大，K/V 重读次数越少（流量小），但片上放不下更多 block（并行度低）。**

---

## 7. FLOPs 账：为什么"多做计算"反而更快

| 项 | 标准 attention | FlashAttention |
|---|---|---|
| 矩阵乘 FLOPs | `4S²D` | `4S²D`（一样） |
| `exp` 次数 | `S²` | `S²`（一样） |
| 额外：每块的 `alpha` 缩放 | 0 | `≈ 2·S²·D/B_n` |
| 额外：反向重算 | 0 | 重新算 `s`（省显存换算力） |

前向的增量主要来自 `acc · alpha` 的逐元素乘，量级是 `S²D/B_n`——**相对 `4S²D` 只有几个百分点**。
所以"FlashAttention 多了很多计算"是夸大，前向几乎是等算力的。

**净效果：算力不变，流量降 2~4×（S=1024）到 ~4×（S=2048），延迟随之下降。**

---

## 8. 与 `w4_attn_naive.cu` 的逐项对照

| 维度 | naive（3 kernel） | FlashAttention（1 kernel） |
|---|---|---|
| kernel 数 | 3（QKᵀ / softmax / PV） | **1** |
| 中间矩阵 | `S_mat`、`P` 各一份 `(B,H,S,S)` 落 HBM | **不落**，只在片上 |
| 显存占用（S=2048） | 2 × 256 MiB = 512 MiB | 0 |
| 屏障 | 每个 kernel 内部都有 | 块内一次性同步，无 kernel 间同步 |
| 瓶颈 | S=1024 时是 L1 扇区 + 指令发射 | 大 S 时是带宽；小 S 时是 matmul 吞吐 |
| 收益来源 | —— | 去掉 4 次 `S²` 往返 + 去掉 kernel 间落盘 |

**w4 的 15.53 ms（S=1024）能到多少？** 流量 272 → 72 MiB（`B_m=128`），若保持同样的有效带宽 18 GB/s，
理论上是 `15.53 × 72/272 ≈ 4.1 ms`。但 Flash 会把瓶颈从"扇区/发射"换成"matmul + 带宽"，
实际大概率落在 **1~2 ms** 区间——**这是 w5 要测出来的数**，不是现在能断言的。

---

## 9. Triton 实现要点（写 w5 时对照）

Triton 里的关键对应关系：

| 数学量 | Triton 代码 |
|---|---|
| Q 块 | `q = tl.load(Q + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd)` |
| `q · kᵀ` | `s = tl.dot(q, tl.trans(k)) * scale` |
| 行最大值 | `m_ij = tl.max(s, axis=1)` |
| 指数 | `p = tl.math.exp2(s * 1.44269504 - m_new * 1.44269504)`（用 `exp2` 更快） |
| 行和 | `l_ij = tl.sum(p, axis=1)` |
| 缩放旧值 | `acc = acc * alpha[:, None]` |
| 累加 | `acc = tl.dot(p.to(tl.float16), v, acc)` |
| 收尾 | `o = acc / l[:, None]` |

**三个实现细节（面试常问）：**

1. **用 `exp2` 而不是 `exp`**：`exp(x) = exp2(x·log2(e))`，把 `log2(e)` 折进 `scale`，省一次乘法。
   Triton 的 `tl.math.exp2` 直接映射到硬件 `MUFU.EX2`。
2. **`p` 要转回 fp16 再喂给 `tl.dot`**：Tensor Core 的 fp16 输入才快；`acc` 保持 fp32 累加。
3. **`num_warps` 与 `num_stages`**：`num_stages` 是软件流水（预取下一个 k/v 块），
   典型 `BLOCK_M=128, BLOCK_N=64, num_warps=8, num_stages=3`。

---

## 10. FlashAttention-2 / -3 改了什么

| 版本 | 核心改进 | 效果（论文口径） |
|---|---|---|
| FA1 (2022) | tiling + online softmax | GPT-2 训练 3× |
| **FA2 (2023)** | ① 减少非 matmul 的 FLOPs（`alpha` 缩放只做必要部分）② 并行度：把"K/V 循环"也切分到多 block（split-K）③ 更好的 warp 分工 | 比 FA1 再快约 2× |
| **FA3 (2024)** | Hopper 专属：warp-specialization（producer/consumer）、异步 TMA、FP8 | H100 上接近理论峰值 |

**面试要点：FA2 的主要贡献不是"更少 IO"，而是"更高 GPU 利用率"**——FA1 在 A100 上只能到 ~35% 峰值算力，
因为一个 block 内的串行 K/V 循环让 SM 闲着。

---

## 11. 常见误区（八条）

1. **"FlashAttention 是近似算法"** —— 错。精确等价，误差只来自 fp16 舍入。
2. **"它减少了 FLOPs"** —— 错。前向 matmul 次数不变，反向甚至要重算。
3. **"K/V 只读一次"** —— 错。读了 `S/B_m` 次。
4. **"任何 S 都提速几十倍"** —— 错。S 小的时候收益只有 2~4×（见 §5）。
5. **"它只对长序列有用"** —— 半对。短序列也省显存，但延迟收益小。
6. **"分块就行，不用 online"** —— 错。不做 online softmax，结果就是错的。
7. **"减 max 只是为了数值稳定，可以省"** —— 错。fp16 下不减必溢出成 inf。
8. **"FlashAttention 一定比 naive 快"** —— 不一定。小 S + 小 D 时，naive 的 kernel 更简单，
   如果 naive 已经写得很高效（扇区 100%、指令效率高），差距会缩小。

---

## 12. 面试问答（10 条，可以直接背）

1. **Q：FlashAttention 解决什么问题？**
   A：attention 的 `S×S` 中间矩阵必须在 HBM 里往返 4 次，S 大时这部分流量是算力的 8 倍。
   FA 用分块 + online softmax 把它留在片上，把 `S²` 项除以块大小。

2. **Q：为什么是精确的？**
   A：softmax 的分子分母可以同时乘 `exp(m' - m)`，比值不变；求和可分段。分块只是改变了累加顺序。

3. **Q：online softmax 的 `alpha` 是什么？**
   A：`exp(m_old - m_new)`，把按旧最大值归一的 `l` 和 `acc` 重新缩放到新基准。

4. **Q：为什么块大小不能太大？**
   A：`q/k/v/acc` 要同时驻留片上，`B_m` 翻倍 → `acc` 翻倍 → 片上放不下更多 block，并行度下降。

5. **Q：反向传播怎么办？**
   A：不存 `S×S`，反向时用 `O` 和 `m/l` 重新算一遍 `s`（重算换显存），这就是"省显存不省算力"。

6. **Q：FA2 比 FA1 强在哪？**
   A：减少非 matmul FLOPs + 把 K/V 循环也并行化，A100 上从 ~35% 提升到 ~70% 峰值。

7. **Q：怎么在 GPU 上判断 FA 是带宽还是算力瓶颈？**
   A：算 AI。S=1024、`B_m=128` 时 AI≈114，仍低于拐点 239.7 → 带宽受限；S 增大、`B_m` 增大后转 compute-bound。

8. **Q：为什么用 `exp2`？**
   A：`exp2` 是单条硬件指令（MUFU.EX2），把 `log2(e)` 折进 `scale` 可省一次逐元素乘。

9. **Q：causal mask 怎么省？**
   A：只在 `j ≤ i` 的块上循环，块内上三角置 `-inf`；对角块要单独处理。工作量约减半。

10. **Q：FlashAttention 和 PagedAttention / KV cache 的关系？**
    A：不同层。FA 优化"一次前向的 attention 计算"；PagedAttention 优化"多请求共享 KV cache 的显存管理"。
    推理引擎里两者叠加使用。

---

## 13. 自测（做完再看答案）

1. S=2048、`B_m=128` 时 K/V 各被读几次？总流量多少？
2. 为什么 `alpha ≤ 1`？如果它大于 1 会发生什么？
3. 把 §3.4 的两个例子用 Python 跑一遍，验证与 `torch.softmax` 一致。
4. 若 `B_m` 从 128 提到 256，流量和并行度各自怎么变？
5. 为什么说"FlashAttention 把 memory-bound 变成 compute-bound"只在 S 足够大时成立？
6. 你的 w4 是 15.53 ms，理论流量比 3.8×，为什么不能说"w5 一定 4.1 ms"？

<details>
<summary>答案要点</summary>

1. 16 次；`16×2×256 KiB = 8 MiB`（单 head），×32 = 272 MiB。
2. `m_new ≥ m_old` → `m_old - m_new ≤ 0` → `exp ≤ 1`。若 >1，说明基准变了但没重新归一，`l`/`acc` 会虚高，结果错。
3. 见 §3.4，两块结果 = 整体结果 = 16.570 / 30.31。
4. 流量减半（K/V 读 8 次），但 `acc` 32 KiB + `q` 16 KiB → 片上只能放 1 个 block，并行度下降。
5. 因为 AI 随 `B_m` 增大而升高，只有跨过 239.7 才转 compute-bound；S 小时 `S²` 项本来就不主导。
6. 因为瓶颈会从"L1 扇区/指令发射"变成"matmul + 带宽"，两者效率不同；且实测有效带宽不是常数。

</details>

---

## 14. 参考文献

- Tri Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, NeurIPS 2022. [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
  （摘要原文：*"an IO-aware exact attention algorithm that uses tiling to reduce the number of memory reads/writes between GPU high bandwidth memory (HBM) and GPU on-chip SRAM"*；报告 GPT-2 训练 3×、BERT-large 端到端 15%、LRA（1K–4K）2.4×）
- Tri Dao, *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*, 2023. [arXiv:2307.08691](https://arxiv.org/abs/2307.08691)
- Triton 官方教程 `06-fused-attention.py`（前向 + 反向的参考实现）
- 本项目内：[`attention入门-从公式到显存流量.md`](attention入门-从公式到显存流量.md)（流量账的入门版）、[`ncu速查.md`](ncu速查.md) §8（算术强度）

---

## 附：校验脚本（复制即可跑）

```python
import torch

def online_softmax_attn(x, v, block):
    # x: (S,), v: (S,)  —— 一维简化版，验证 online 递推
    m, l, acc = float("-inf"), 0.0, 0.0
    for start in range(0, len(x), block):
        s = x[start:start + block]
        vv = v[start:start + block]
        m_new = max(m, float(s.max()))
        alpha = torch.exp(torch.tensor(m - m_new))
        p = torch.exp(s - m_new) if m_new > float("-inf") else torch.exp(s - s.max())
        l = l * float(alpha) + float(p.sum())
        acc = acc * float(alpha) + float((p * vv).sum())
        m = m_new
    return acc / l

x = torch.tensor([2.0, 0.0, 1.0, -1.0])
v = torch.tensor([10.0, 20.0, 30.0, 40.0])
ref = float((torch.softmax(x, dim=-1) * v).sum())
print("block=2:", online_softmax_attn(x, v, 2), "ref:", ref)
print("block=1:", online_softmax_attn(x, v, 1), "ref:", ref)
```

> 预期两行都打印 ~16.570（fp32 精度内）。**这就是 online softmax 的全部正确性证据。**
