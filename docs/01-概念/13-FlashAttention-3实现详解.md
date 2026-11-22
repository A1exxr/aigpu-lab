# FlashAttention-3 实现详解（Hopper 专属）

> **这份文档写给谁**：写给已经手写完 FA2 前向（`w12c` → `w12d` → `w12e` → `w12f` → `w12g`，4060 上 2.61 → 11.5 TFLOPS）的人。
> 所以第 1–3 章只补「你还没亲手碰过的东西」，第 4 章起才是 FA3 真正的增量。
> 所有公式按 Markdown + LaTeX 写（`$...$` / `$$...$$`），代码按 CUDA/PTX 原样给，能直接对照仓库。
>
> 结论先行：**FA3 不是新算法，是把 FA2 的同一套数学塞进 Hopper 的异步硬件里**。数学一个字没改，改的是「谁在等谁」。

---

## 0. 一页速览

| 项目 | FlashAttention-2 | FlashAttention-3 |
|---|---|---|
| 目标硬件 | Ampere / Ada（`sm_80` / `sm_89`） | **Hopper（`sm_90a`）** |
| H100 上 FP16 前向 | ~350 TFLOPS（**35%** 峰值） | **最高 740 TFLOPS（75%）** |
| 相对加速 | 1.0× | **1.5–2.0×** |
| FP8 | 无 | 接近 **1.2 PFLOPS**，误差比 baseline FP8 小 **2.6×** |
| 三个关键技术 | — | ① warp specialization（异步搬运）② GEMM 与 softmax 交错（pingpong + 2-stage）③ FP8 block quantization + 非相干处理 |
| 论文 | [arXiv:2307.08691](https://arxiv.org/abs/2307.08691) | [arXiv:2407.08608](https://arxiv.org/abs/2407.08608) |
| 官方博客 | — | [tridao.me/blog/2024/flash3](https://tridao.me/blog/2024/flash3/) |

H100 SXM5 的出厂数字（后面所有百分比的分母）：

$$
\text{FP16 TC} = 989\ \text{TFLOPS},\qquad
\text{FP8 TC} = 1978\ \text{TFLOPS},\qquad
\text{SFU（exp 等）} \approx 3.9\ \text{TFLOPS}
$$

---

## 1. 从 350 到 740：钱花在哪四级台阶

FA3 的加速不是一步到位的，博客里给了四级台阶，每一级对应一类代码改动。**记住这张表，面试问到「FA3 为什么快」时按级回答，不要一句话糊过去。**

| 级 | 做了什么 | H100 FP16 前向 | 增量 | 你现在有吗 |
|---|---|---|---|---|
| 0 | FA2 原版（`mma.sync` + `cp.async`） | ~350 TFLOPS（35%） | — | ✅ `w12f` 就是这个思路 |
| 1 | **只换指令**：TMA 搬数据 + `wgmma` 算 GEMM | 540–570 TFLOPS | +55~63% | ✅ `w12e/w12f`（QK 用 wgmma、K/V 用 TMA） |
| 2 | **pingpong 调度**（warpgroup 间交替抢 GEMM/softmax） | ~620 TFLOPS | +9% | ❌ |
| 3 | **intra-warpgroup 2-stage**（同一个 warpgroup 内 GEMM 与 softmax 交错） | 640–660 TFLOPS | +3~6% | ❌ |
| 4 | 再往上（FP8、persistent、varlen、in-kernel transpose） | 740（FP16 上限）/ 1.2 PFLOPS（FP8） | — | ❌ |

**两个反直觉的点，先记住**：

1. **第 1 级（换指令）拿走了大部分收益**。原因很硬：`mma.sync` 只能跑到 Hopper Tensor Core 峰值的约 **2/3**（见 [arXiv:2402.13499](https://arxiv.org/abs/2402.13499)），不换 `wgmma` 天花板就锁死在 66%。
2. **第 2、3 级收益只有 10% 左右，但难度陡增**。它们解决的是「Tensor Core 在算的时候，SFU 在发呆」——而这个浪费最多占 50%（下一节推导），所以理论收益上限就是那么多。

---

## 2. 硬件前提：Hopper 的四个新东西

### 2.1 TMA（Tensor Memory Accelerator）

**一句话**：一块专门的搬运硬件，你给它一张「描述符」（描述多维张量的形状、步长、box 大小、swizzle 模式），它在设备端按坐标搬一整块，**不占寄存器、不占线程**。

对比你熟悉的 `cp.async`：

| | `cp.async` | TMA（`cp.async.bulk.tensor`） |
|---|---|---|
| 谁算地址 | **每个线程自己算** | 硬件按描述符算 |
| 越界处理 | 自己写 `if` | 硬件按 box 边界**零填充** |
| 寄存器占用 | 每个线程要留地址寄存器 | 只有发起线程要 1 个描述符指针 |
| 布局能力 | 任意重排（每线程一个地址） | 线性 / 32-64-128 B swizzle |
| 完成通知 | 线程级 `commit_group` / `wait_group` | **只能上报给 mbarrier 的 transaction 计数** |
| SASS 助记符 | `LDGSTS.E.BYPASS.128` | `UTMALDG.2D` / `UTMALDG.3D` |

我们 `w12f_fa2_tma.cu` 里已经用 3D 描述符搬 KV（`gdim={D,S,BH}`），这是必要的：KV 的物理形状是 `[B·Hkv][S][D]`，用 2D 就得把 `bh` 烤进行坐标，`S=125` 这种非整除 case 会读到下一个头的 KV（**不崩、数值错**）。

host 端建描述符（我们代码的简化版）：

```cpp
CUtensorMap tmap;
uint64_t gdim[3]    = {D, S, B*Hkv};          // 最内维在前
uint64_t gstride[2] = {D * sizeof(half),       // 第 1 维步长（字节）
                       D * S * sizeof(half)};  // 第 2 维步长
uint32_t box[3]     = {D, TILE_N, 1};          // 一次搬 [1, TILE_N, D]
cuTensorMapEncodeTiled(&tmap, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 3,
                       ptr, gdim, gstride, box,
                       CU_TENSOR_MAP_INTERLEAVE_NONE,
                       CU_TENSOR_MAP_SWIZZLE_128B,   // ← 关键
                       CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                       CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
```

**三条硬约束**（踩过就知道）：

1. 描述符必须 **64 B 对齐**，且只能通过 `__grid_constant__ const CUtensorMap` 传参给 kernel（不能在 device 端改基址）。
2. 最内维的 box 字节数必须等于 swizzle 粒度：`SWIZZLE_128B` ⇒ 最内维必须是 128 B（`D=64` 的 `half` 正好 128 B ✅）。
3. `globalStrides` 只有 `rank−1` 项，**单位是字节**，且必须是 16 的倍数。

### 2.2 WGMMA

**一句话**：一个 **warpgroup（128 线程）** 一起发一条矩阵乘，A/B 可以直接从 **shared memory** 读（SS 形式），累加器留在寄存器。

| | `mma.sync`（Ampere/Ada） | `wgmma`（Hopper） |
|---|---|---|
| 发射单位 | 1 个 warp（32 lane） | **1 个 warpgroup（4 warp = 128 lane）** |
| 形状 | `m16n8k16` | `m64nNk16`（N = 8..256） |
| A/B 来源 | 只能寄存器（要先 `ldmatrix`） | **smem（SS）或寄存器（RS）** |
| 同步性 | 同步指令 | **异步**：`fence` → `commit_group` → `wait_group` |
| 峰值 | 约 Hopper 的 2/3 | 100% |
| SASS 助记符 | `HMMA.16816.F32` | **`HGMMA.64x32x16.F32`**（注意不叫 WGMMA） |

**描述符（descriptor）位域**——这是 wgmma 最容易写错的地方：

```
 bit 63      62     61..46      45..32      31..30   29..16      15..0
┌────────┬────────┬───────────┬───────────┬────────┬───────────┬────────┐
│ 0      │ swizzle│   base>>4 │   SBO>>4  │  0     │   LBO>>4  │ addr>>4│
└────────┴────────┴───────────┴───────────┴────────┴───────────┴────────┘
```

- `addr>>4`：起始地址（**只有 14 位**，所以 smem 地址 16 B 对齐后能表示 256 KB）
- `LBO`：leading byte offset，**块内层**（K 方向）跨一个 8 列块的距离
- `SBO`：stride byte offset，**块外层**（M/N 方向）跨一个 8 行块的距离
- `swizzle`：0 = 无（8 行压实 128 B 的 core-matrix 布局）、1/2/3 = 128/64/32 B 行主序 + 异或

代码（我们 `w12f` 的写法）：

```cpp
__device__ __forceinline__ unsigned long long make_desc(
        const void* p, unsigned lbo, unsigned sbo, unsigned swz = 0) {
    unsigned long long addr = (unsigned long long)__cvta_generic_to_shared(p);
    return ((addr & 0x3FFFFULL) >> 4)
         | ((unsigned long long)(lbo >> 4) << 16)
         | ((unsigned long long)(sbo >> 4) << 32)
         | ((unsigned long long)swz << 62);
}
```

> **为什么参数类型一个是 `uint32_t` 一个是 `unsigned long long`？**
> 当地址**单独当操作数**用（`mbarrier.init.shared.b64 [%0]`、`ldmatrix`、`cp.async`）⇒ PTX 约束是 `"r"`（32 位，smem 用 32 位寻址）；
> 当它被**打包成描述符当操作数**传（`wgmma`）⇒ 约束是 `"l"`（64 位整数）。
> 不是「地址 32 位 / 数据 64 位」——描述符里也有地址，只是被压进低 14 位。

**两条必记的 wgmma 同步规则**：

```cpp
wgmma.fence.sync.aligned;              // 第一条 wgmma 之前必须有一次
wgmma.mma_async... ;                   // 若干条
wgmma.commit_group.sync.aligned;       // 成组提交
wgmma.wait_group.sync.aligned 0;       // 0 = 等全部完成（读累加器前必须）
```

`wgmma.fence` 和 `fence.proxy.async.shared::cta` **不是一回事**：
- `wgmma.fence` 是 wgmma 专用的顺序栅栏（保证之前的 smem/寄存器写对后面的 wgmma 可见）；
- `fence.proxy.async` 是 **generic proxy ↔ async proxy** 的可见性发布，只在「线程写 smem → TMA 读」或「TMA 写 → 普通 `ld` 读」时需要。
我们的 K/V 是 TMA 写、wgmma 读，`mbar_wait` 已经把数据可见性管住了，两条都不用加。

### 2.3 异步模型与两个 proxy

Hopper 上同一个 SM 里有**两套内存访问通道**：

```
        ┌──────────── 线程（generic proxy）────────────┐
        │  ld.global / st.shared / ldmatrix / ld.shared │
        └───────────────────┬──────────────────────────┘
                            │  fence.proxy.async 才能互相看见
        ┌───────────────────┴──────────────────────────┐
        │  异步单元（async proxy）：TMA、wgmma、cp.async │
        └──────────────────────────────────────────────┘
```

由此得到一张「谁负责什么」的表——**这张表能解释 90% 的同步 bug**：

| 你要保证的事 | 用什么 |
|---|---|
| 线程到齐了 | `__syncthreads()` / `bar.sync` |
| 异步搬运的数据到了 | `mbarrier` 的 transaction count 归零（TMA）/ `cp.async.wait_group` |
| wgmma 算完了 | `wgmma.commit_group` + `wgmma.wait_group N` |
| 普通写对 async 单元可见 | `fence.proxy.async.shared::cta` |
| smem/寄存器写对 wgmma 可见 | `wgmma.fence.sync.aligned` |

### 2.4 `setmaxnreg` 与寄存器预算

`setmaxnreg.inc/dec.sync.aligned.u32 N` 让一个 **warpgroup** 动态把自己的寄存器上限改掉，把省下来的借给别的 warpgroup。

- 立即数范围 **24–256**，且必须是 **8 的倍数**（写 16 报 `value '16' out of range`）。
- 用途：producer warpgroup 只发 TMA，用不到多少寄存器 → `dec 40`；consumer 要装 64×128 的累加器 → `inc 232`。
- **实测提醒**：我们三组最小复现在 CUDA 12.8 / `sm_90a` 上 **SASS 里 `SETMAXNREG` 恒为 0**，ptxas 把它优化掉了。原因很可能是 kernel 只用了 102 个寄存器（< `65536/384 = 170`），本来就不缺，编译器判定无收益。**所以不要用「SASS 里有没有 SETMAXNREG」当验收判据**，要验就验结构（`LDGSTS=0` + `UTMALDG≥2` + `ARRIVE/DEPBAR` 计数）。

---

## 3. 数学：FA 在算什么（能写代码的精度）

### 3.1 定义

对单个 head、单个 batch：

$$
S = \frac{QK^\top}{\sqrt{d}} \in \mathbb{R}^{N\times N},\qquad
P = \operatorname{softmax}(S)\ (\text{按行}),\qquad
O = PV \in \mathbb{R}^{N\times d}
$$

数值稳定的 softmax：

$$
P_{ij} = \frac{e^{S_{ij} - m_i}}{\ell_i},\qquad
m_i = \max_j S_{ij},\qquad
\ell_i = \sum_j e^{S_{ij} - m_i}
$$

### 3.2 分块 online softmax（FA1/2/3 共用）

把 K/V 沿列方向切成 `Bc` 大小的块，逐块累积。设已处理完 $j < j_0$ 的块，持有 $(m, \ell, O)$，来新块 $S^{(j_0)}$：

$$
\tilde m = \max_j S^{(j_0)}_{ij},\qquad
m' = \max(m, \tilde m)
$$

$$
\alpha = e^{m - m'},\qquad
\ell' = \alpha\,\ell + \sum_j e^{S^{(j_0)}_{ij} - m'}
$$

$$
O' = \alpha\, O + \sum_j e^{S^{(j_0)}_{ij} - m'}\, V^{(j_0)}_{j}
$$

最后 $O_{\text{final}} = O / \ell$。

**三条容易写错的**（也是我们错题本 E-107/E-108 的内容）：

1. $\tilde m$ 必须是**整块**的 max，不是逐元素的 `fmaxf(m, s[j])`；
2. $\alpha = e^{m - m'}$ 的方向不能反（必须 $\le 1$，反了会指数放大、不报错只爆数值）；
3. $O$ 是**向量**，乘法是逐元素的 `acc[d] *= alpha`，不是把 `acc` 整体赋值。

### 3.3 反向（了解即可，FA3 的反向也是这套）

$$
\delta_i = \sum_d dO_{id} O_{id},\qquad
dS = P \odot \left(dP - \delta\right),\qquad
dP = dO\,V^\top
$$

$$
dQ = \frac{dS\,K}{\sqrt d},\qquad
dK = \frac{dS^\top Q}{\sqrt d},\qquad
dV = P^\top dO
$$

FA3 反向的工程难点与正向同源：`dS` 要重算 `S`（recompute），`dV/dK` 要跨块累加（原子加或两趟 kernel）。

### 3.4 FLOPs 与字节口径

前向（非 causal）：

$$
\text{FLOPs} = 4 N^2 d \quad(\text{QK}^\top: 2N^2d,\ PV: 2N^2d)
$$

causal（下三角）：

$$
\text{FLOPs}_{\text{causal}} = 2 N (N+1) d \approx \tfrac12 \cdot 4N^2 d
$$

> **口径纪律**：写简历/报告时 causal 必须减半，否则虚高 $2N/(N+1) \approx 2\times$。我们 `w12c/w12d/w12e` 用的都是 $2BHS(S+1)D$。

---

## 4. FA3 的三招

### 4.1 为什么 softmax 值得抢：SFU 只有 3.9 TFLOPS

这是 FA3 全部动机的定量依据，**必须自己会算**。

H100 SXM5 的 SFU（特殊函数单元，`exp` 走它）吞吐是 **每 SM 每 clock 16 次**操作：

$$
16 \times 132\ \text{SM} \times 1.83\ \text{GHz} = 3.87\ \text{TFLOPS}
$$

而 FP16 Tensor Core 是 989 TFLOPS，比值：

$$
\frac{989}{3.87} \approx 256
$$

再看每个 $(i,j)$ 元素上，matmul 与 exp 的工作量比。设 head dim 为 $d$：$S_{ij}$ 需要 $2d$ FLOP（点积），再用它算 $PV$ 又摊到 $2d$ FLOP，合计 $4d$ FLOP / 1 次 exp。所以

$$
\frac{\text{matmul FLOPs}}{\text{exp 次数}} = 4d
\;\xrightarrow{d=128}\; 512
$$

于是时间比：

$$
\frac{T_{\exp}}{T_{\text{matmul}}} = \frac{512}{256} \approx 2.0
$$

**结论：如果串行执行，exp 要花掉 matmul 一半的时间。** 这也是为什么博客说「head dim 128 时 exp 可能占 50%」。FP8 下 matmul 再快一倍，这个比例变成 100%——**FP8 的主要矛盾不是精度，是 softmax 跟不上**。

> 我们自己机器的对照（同样的算法，换成 4060 的数）：FP16 TC 峰值 65.2 TFLOPS，`MUFU.EX2` 每 SM 16 lane/clk × 24 SM × 2.655 GHz ≈ **1.02 TFLOPS**，比值约 **64×**。而我们 `w12` 的 head dim 是 64，工作量比只有 $4d = 256$，所以
> $$
> \frac{T_{\exp}}{T_{\text{matmul}}} = \frac{256}{64} \approx 0.25
> $$
> 即 exp 至少占 matmul 时间的 **25%**（head dim 128 时就是 50%）。这就是 w12f 里「wgmma 已经把 QK 算完了，但周期里仍有一段吃不掉」的来源：Tensor Core 在等 SFU。

### 4.2 招一：warp specialization（producer / consumer）

**结构**（我们 `w12g` 已经是这个骨架）：

```
SM
├── warpgroup 0  ← producer：只发 TMA，把 K/V tile 搬进 smem
│                  setmaxnreg.dec 40（不需要那么多寄存器）
└── warpgroup 1,2 ← consumer：只算 wgmma + softmax + PV
                   setmaxnreg.inc 232（累加器要吃寄存器）
```

**握手靠两个 mbarrier 数组**（双缓冲）：

| barrier | 谁 arrive | 谁 wait | 计数 |
|---|---|---|---|
| `full[i]` | TMA 的 transaction 字节数（`arrive_expect_tx`） | consumer | 1 次到达（字节归零才翻相位） |
| `empty[i]` | 每个 consumer warp 各一次（`lane==0`） | producer | `CONS_WARPS`（= consumer warp 总数） |

关键代码（`w12g_fa2_ws.cu` 的结构）：

```cpp
// producer（wg0）
if (tid == 0) {
    mbar_arrive_expect_tx(&full[buf], 2 * TILE_N * HEAD_DIM * sizeof(half)); // K+V
    tma_load_3d(Kbuf[buf], ..., &kmap, ...);
    tma_load_3d(Vbuf[buf], ..., &vmap, ...);
}

// consumer（wg1/2）
mbar_wait(&full[buf], (jt >> 1) & 1);       // 等数据到齐（相位 = jt/2 奇偶）
if (jt <= jt_end_wg) { /* wgmma + softmax + PV */ }
if (lane == 0) mbar_arrive(&empty[buf]);    // 必须在 if 外面！

// producer 复用 buf 之前
if (jt >= 2) mbar_wait(&empty[buf], ((jt - 2) >> 1) & 1);
```

**为什么 `jt >= 2`、相位为什么是 `((jt-2)>>1)&1`**：双缓冲下 `jt=0/1` 是灌水，没有「消费者用完」可等；从 `jt=2` 起复用 `buf`，等的是 `jt-2` 那轮的签退，而 `empty[buf]` 每两轮才翻一次相位。

**为什么空转的 consumer 也要 arrive**：若把 `mbar_arrive` 挪进 `if (jt <= jt_end_wg)`，`S=125` 这种 `jt_end_wg < jt_end` 的 case 里，producer 的计数永远到不了 `CONS_WARPS` ⇒ **死锁**。

**这一招的收益**：博客把「warp specialization 单独用」归到第 1 级台阶里（350 → 540–570 TFLOPS 一起算），因为 wgmma/TMA 一上就必须拆 producer/consumer，否则数据流对不上。

### 4.3 招二：pingpong 调度（warpgroup 间交替）

**问题**：即使拆了 producer/consumer，两个 consumer warpgroup 仍然是「同时算 GEMM，然后同时卡在 softmax」。Tensor Core 与 SFU 交替空闲。

**做法**：用 `bar.sync` 把两个 consumer warpgroup 的相位**错开半拍**：

```
时间 →
wg1: [GEMM j ][GEMM j+1][ softmax j+1 ][GEMM j+2]...
wg2: [ softmax j ][GEMM j+1][ softmax j+1 ]...
                ↑ 一个在算 GEMM 时，另一个在做 softmax
```

用命名 barrier（`bar.sync 1, 256`）或 mbarrier 实现「wg1 先跑、wg2 等半拍」。博客给的实测：**570 → 620 TFLOPS**（head dim 128，seqlen 8K）。

**为什么只有 9%**：能重叠的上限就是 softmax 的占比；而 pingpong 让寄存器压力上升（两个 warpgroup 都要留累加器），占用率会掉一点，两头相抵。

### 4.4 招三：intra-warpgroup 2-stage（同一 warpgroup 内交错）

即使只有一个 warpgroup，也能让它内部的 **GEMM(i) 与 softmax(i−1) 同时进行**：

```
wgmma(Sc[buf0])  ────────────────┐
                                 ├── 同一 warpgroup 的两条独立依赖链
softmax(Sc[buf1]) ───────────────┘
```

条件：**两份 S 累加器**（`Sc[2][...]`），一份在算 wgmma、一份在做 softmax。博客实测 **620 → 640–660 TFLOPS**，代价是寄存器压力上升（要同时装两份 S 与 softmax 的输入输出）。

伪代码：

```cpp
// 需要 2 份 S 累加器 + 2 份 m/l（先算 tile 0，再进稳态循环）
compute_qk(Sc[0], kv[0]);
for (int j = 1; j <= jt_end; ++j) {
    if (tid == 0) issue_tma(j);
    compute_qk(Sc[j & 1], kv[j & 1]);     // wgmma 异步发射，不等
    softmax_update(Sc[(j - 1) & 1]);      // 与上面那条 wgmma 重叠
    pv_accumulate(Sc[(j - 1) & 1]);       // 同一个 tile 的 PV
    wgmma_wait_group(1);                  // 只等「除最后一条外」的 wgmma
}
```

注意最后是 `wait_group 1` 而不是 `0`：留下最新那条在飞，才能和 softmax 重叠。

---

## 5. FP8：block quantization + 非相干处理

> 先说结论：**我们这台 4060（Ada）没有 FP8 Tensor Core，H20 有但 FP8 收益需要 Hopper 级调优，所以这一章只做「看得懂、答得出」，不写代码。**

### 5.1 两种 FP8 与缩放

| 格式 | 指数位 | 尾数位 | 范围 | 用途 |
|---|---|---|---|---|
| E4M3 | 4 | 3 | ±448 | **前向的 Q/K/V/P** |
| E5M2 | 5 | 2 | ±57344 | 反向（动态范围大） |

量化公式（per-tensor 版）：

$$
s = \frac{\max |x|}{448},\qquad \tilde x = \operatorname{round}\!\left(\frac{x}{s}\right),\qquad x \approx s\,\tilde x
$$

**block quantization**：per-tensor 会被离群值毁掉（激活里 0.1% 的元素可能比其余大 100×）。所以沿 K 方向按 block（如 128 列）算一个 scale，softmax 前把 `P` 再按行量化一次。

### 5.2 为什么 QK^T 的缩放可以「分开算」

因为 scale 是标量、乘法可交换：

$$
\tilde Q \tilde K^\top \cdot (s_Q s_K) \approx Q K^\top
$$

所以 Q 和 K 各自量化、各自带 scale，GEMM 用 FP8、结果乘回 $s_Q s_K$ 即可。**PV 同理**，但 P 的 scale 要按行算（`P` 每行和固定为 1，scale 很小）。

### 5.3 非相干处理（incoherent processing）

**动机**：离群值让量化误差爆炸。

**做法**：乘一个随机符号的 Hadamard 矩阵 $H$（正交、$HH^\top = I$）把离群值「摊开」：

$$
Q' = QH,\quad K' = KH \;\Longrightarrow\;
Q'K'^\top = Q H H^\top K^\top = QK^\top
$$

**数学上完全等价，但每个元素的动态范围被压平了**，量化误差随之下降。$H$ 是 Hadamard 矩阵，所以 $QH$ 可以按 $O(d \log d)$ 的快速变换做（不是 $O(d^2)$ 矩阵乘），而且这个变换是**访存受限**的，可以和 RoPE 融合「免费」做掉。

博客的实测：模拟 0.1% 元素为离群值时，**量化误差降到 1/2.6**。论文里 2.6× 这个数就是从这里来的。

### 5.4 FP8 还要注意的三件事

1. **in-kernel transpose**：`wgmma` 要求 B 是列主序（或 K-major），FP8 下没法用 `ldmatrix.trans`（它只认 16 位），必须在 kernel 里做一次转置 → 这是论文提到的额外优化点。
2. **累加器仍是 FP32**：FP8 只影响输入，`Sc`/`O` 必须 FP32，否则误差爆炸。
3. **不要用 FP8 算 softmax 的输入**：`S` 要反量化回 FP16/FP32 再做 exp。

---

## 6. 代码结构（CUTLASS 3.x 路线）

FA3 是**用 CUTLASS 3.x 的组件拼出来的**，不是从零手写 PTX。理解它的分层比读懂每个模板参数更重要：

```
flash_fwd_kernel.h                 ← kernel 入口，选 tile 形状、跑主循环
├── CollectiveMma                 ← QK^T 的 wgmma 封装（SS 形式）
│   ├── TiledMma                  ← wgmma 的 tile 形状/描述符
│   └── MainloopPipeline          ← TMA → smem 的多 stage 流水（mbarrier 数组）
├── CollectiveSoftmax             ← mask / 行 max / exp / 行和
├── CollectiveEpilogue            ← P·V（第二个 wgmma）+ 除 l + 写回
└── tile_scheduler                ← 静态/持久化调度，支持 varlen
```

**主循环（伪代码，与我们的 `w12f` 同构）**：

```cpp
// 1) prologue：Q 进 smem（一次），TMA 发第 0 块 K/V
// 2) 稳态循环
for (int jt = 0; jt <= jt_end; ++jt) {
    pipeline.consumer_wait(jt);              // mbarrier 等数据
    gemm_qk(Sc, Qs, Ks[jt & 1]);             // wgmma SS，异步
    wgmma.wait_group(0);
    softmax_mask(Sc, causal, row_max, l);    // SFU：exp
    gemm_pv(O, Sc_as_A, Vs[jt & 1]);         // 第二个 wgmma（或 mma.sync）
    pipeline.consumer_release(jt);           // 通知 producer 可覆盖
}
// 3) epilogue：O / l，写回 HBM
```

**三个与我们不同的工程点**：

1. **persistent kernel**：grid 固定为 SM 数（× occupancy），tile 由 scheduler 内部分发。好处是省 launch 与让 tile 的 K/V 复用率受控。我们是「一个 block 一个 Q tile」的朴素调度。
2. **varlen（变长序列）**：`cu_seqlens` 传进来，scheduler 按请求切 tile，避免 padding 浪费。我们是定长 `S`。
3. **in-kernel transpose**：FP8 需要（见 5.4）。

---

## 7. 你手上这版（w12f / w12g）离 FA3 还有多远

| FA3 的组成 | 我们的实现 | 状态 |
|---|---|---|
| TMA 搬 K/V | `w12f`：3D 描述符 + `SWIZZLE_128B`，`UTMALDG.3D ×4` | ✅ |
| mbarrier 双缓冲 + 相位 | `w12f`：K/V 共用一条 `full`（`expect_tx` 8192 B），相位 `(jt>>1)&1` | ✅ |
| wgmma 算 QK | `w12e/w12f`：`wgmma.m64n32k16`，`HGMMA ×4` | ✅ |
| **wgmma 算 PV** | 仍是 `mma.sync`（P 在寄存器，走 RS 形式才能上 wgmma） | ❌ |
| warp specialization | `w12g`：1 producer + 2 consumer，`LDGSTS=0`、`ARRIVE 8` | ✅ 结构有，**收益没验**（`SETMAXNREG` 被 ptxas 优化掉） |
| pingpong 调度 | 无（两个 consumer 同相位） | ❌ |
| intra-warpgroup 2-stage | 无（单份 `Sc`） | ❌ |
| FP8 + 非相干处理 | 无（Ada 无 FP8 TC） | ❌（不做） |
| persistent kernel | 无（grid = `ceil(S/128) × B × H`） | ❌ |
| varlen | 无（定长 `S`） | ❌ |

### 7.1 逐项补齐的代价（按性价比排序）

| 顺序 | 任务 | 工作量 | 判据 | 在 H20 上能验吗 |
|---|---|---|---|---|
| 1 | **wgmma 算 PV**（RS 形式，A 从寄存器） | 1 天 | `HMMA.16816` 计数 → 0，`HGMMA` 计数翻倍 | ✅ 能，且是纯收益（省掉每 warp 重复读整块 V 的 8× smem 放大） |
| 2 | pingpong 调度 | 0.5–1 天 | 两个 warpgroup 的 `bar.sync` 错开半拍；TFLOPS 提升 | ✅ 能 |
| 3 | intra-warpgroup 2-stage | 1 天 | 两份 `Sc`；寄存器不溢出（`-Xptxas -v`） | ✅ 能 |
| 4 | persistent kernel | 0.5 天 | grid = 常数；小 S 时启动开销占比下降 | ✅ 能 |
| 5 | varlen | 1 天 | 两个不等长请求拼一个 batch，数值与逐个算一致 | ✅ 能 |
| 6 | FP8 | 3–5 天 | 误差 < 1e-2 且 TFLOPS 翻倍 | ⚠️ H20 有 FP8 TC，但没有 FA3 级调优经验，收益不确定 |

### 7.2 这趟 H20 只验「正确性 + 结构」

`H20` 是 Hopper（CC 9.0），`wgmma`/TMA/`setmaxnreg` 都支持，但 FP16 算力只有 H100 的约 1/7。所以：

- **能验**：三档 `max|diff|` 是否与 4060 上 `w12e` 位级一致、`HGMMA`/`UTMALDG` 计数对不对、有没有死锁。
- **不能验**：TFLOPS 绝对值、与 SDPA 的 ratio（H20 上 SDPA 自己也掉）、pingpong 的收益幅度（小算力下 softmax 占比更高，方向会变）。

---

## 8. 12 个高频坑

1. **`sm_90` 而不是 `sm_90a`**：`wgmma`/`setmaxnreg` 只在 `sm_90a` 下生成，写 `sm_90` 能编过但指令退化。
2. **描述符 64 B 对齐**：不对齐 `cuTensorMapEncodeTiled` 直接返回错误码。
3. **`boxDim` 最内维 ≠ swizzle 粒度**：`SWIZZLE_128B` 要求最内维 128 B。
4. **TMA 越界是零填充**：所以 `S=125` 不会崩，只会静默算出错的数——必须自己写 `max|diff|` 对拍。
5. **`mbarrier` 相位算错**：`(jt>>1)&1` 与 `((jt-2)>>1)&1` 混用 ⇒ 卡死或读到上一轮数据。
6. **空转 warp 不 arrive** ⇒ 死锁（见 4.2）。
7. **`wgmma.fence` 与 `fence.proxy.async` 混用**：前者管 wgmma 顺序，后者管 proxy 可见性。
8. **`wait_group 0` 写死**：想重叠必须留 1 组在飞（`wait_group 1`）。
9. **`swizzle=0` 与 `swizzle=1` 的描述符数值不通用**：前者是 8 行压实 128 B，后者是行主序 + 异或；换布局必须重推 LBO/SBO。
10. **`ldmatrix` 不懂 swizzle**：软件要自己算 `swz(r,c)`；而 wgmma 用描述符自己解。
11. **累加器在寄存器但写法导致的溢出**：`Sc[TILE_N/8][4]` 是 16 个 float，`acc[HEAD_DIM/8][4]` 是 32 个；超过 100 个 float/线程就开始 spill。
12. **把 `__syncthreads()` 当万能同步**：它只保证 CTA 内线程到齐，不保证 async 单元的数据可见。

---

## 9. 面试问答（15 条）

1. **FA3 相对 FA2 改了什么？** 数学没改，改的是执行：TMA/wgmma 异步化、warp specialization、GEMM 与 softmax 交错（pingpong + 2-stage）、FP8 量化。
2. **为什么 H100 上 FA2 只有 35%？** `mma.sync` 只能到 Hopper TC 峰值的 ~2/3；且 softmax 与 GEMM 串行，Tensor Core 与 SFU 互相等。
3. **FA3 fp16 到多少？** 最高 740 TFLOPS = 75% 峰值，比 FA2 快 1.5–2.0×。
4. **FP8 呢？** 接近 1.2 PFLOPS，误差比 baseline FP8 小 2.6×。
5. **为什么 softmax 这么贵？** SFU 只有 3.9 TFLOPS（16 ops/SM/clk × 132 SM × 1.83 GHz），比 matmul 低 256×；head dim 128 时 matmul 与 exp 的工作量比是 512，所以串行时 exp 占 matmul 一半时间。
6. **pingpong 怎么实现？** 两个 consumer warpgroup 用 `bar.sync` 错开半拍，A 做 GEMM 时 B 做 softmax。
7. **2-stage 又是什么？** 同一 warpgroup 内两份 S 累加器，wgmma 算第 j 块时对第 j−1 块做 softmax，靠 `wait_group 1` 留一组在飞。
8. **它们的收益？** 570 → 620 → 640–660 TFLOPS，约 10%；再往上要 FP8。
9. **warp specialization 的两种 warpgroup 怎么分工？** producer 只发 TMA（`setmaxnreg.dec`），consumer 只算 wgmma/softmax（`inc`），用 `full`/`empty` 两组 mbarrier 双缓冲握手。
10. **`setmaxnreg` 有什么用？** 动态调 warpgroup 的寄存器上限，把 producer 省下的借给 consumer 装累加器（24–256，8 的倍数）。
11. **wgmma 的形状？** `m64n8..256k16`，一个 warpgroup（128 lane）发一条；SASS 里叫 `HGMMA`。
12. **TMA 比 cp.async 强在哪？** 地址由硬件算、越界零填充、不占寄存器；只能线性/32-64-128B swizzle，产不出任意重排。
13. **FP8 为什么用非相干处理？** 离群值让 per-tensor 量化误差大；乘随机符号 Hadamard 矩阵把动态范围摊平，$QH(HK)^\top = QK^\top$ 数学不变，误差降 2.6×。
14. **FA3 在 A100 上能跑吗？** 不能，`wgmma`/TMA/FP8 TC 都是 Hopper 才有；A100 只能跑 FA2。
15. **你做过哪一版？** 手写 FA2 前向（causal + GQA），4060 上从标量 2.61 TFLOPS 做到 `mma.sync`+swizzle+cp.async 11.5 TFLOPS；再把 QK 换 `wgmma`、K/V 换 TMA 写成 Hopper 版，SASS 验到 `HGMMA`/`UTMALDG` 真生成，数值待 H100/H20 上机。

---

## 10. 参考

- 论文：[FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608)（Shah, Bikshandi, Zhang, Thakkar, Ramani, Dao, 2024）
- 博客：[FlashAttention-3 官方解读](https://tridao.me/blog/2024/flash3/)
- 代码：[Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)（`hopper/` 目录，CUTLASS 3.x）
- FA2：[arXiv:2307.08691](https://arxiv.org/abs/2307.08691)
- wgmma 只能到 2/3 峰值的证据：[arXiv:2402.13499](https://arxiv.org/abs/2402.13499)
- Hopper 白皮书：[NVIDIA H100 Tensor Core GPU Architecture](https://resources.nvidia.com/en-us-tensor-core/gtc22-whitepaper-hopper)
- 非相干处理的来源：[QuIP](https://arxiv.org/abs/2307.13304) / [QuIP#](https://arxiv.org/abs/2402.04396)
- 我们自己的对照实现：`kernels/w12e_fa2_wgmma.cu`（QK=wgmma）、`kernels/w12f_fa2_tma.cu`（K/V 全 TMA）、`kernels/w12g_fa2_ws.cu`（warp specialization）
