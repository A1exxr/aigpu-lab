# w7 PagedAttention decode —— 实验过程与结论

> 手写 CUDA，从零实现 PagedAttention 的 decode kernel，在 RTX 4060 Laptop 上从 2.36 ms 优化到 1.25 ms。
> 本文记录每一步改了什么、为什么快、以及为什么停在这里。
> 数据来源：`results/kernels.md`（每次覆盖）+ 本文表格（手动维护，留历史）。

---

## 1. 问题

decode 阶段每个请求只有 **1 个 query token**，但要和 **2048 个历史 token** 的 KV cache 做 attention。
KV cache 存在固定大小的物理块里，不连续；靠一张 `block_table` 做逻辑块 → 物理块的映射。这就是 vLLM 的 PagedAttention。

| 项 | 值 |
|---|---|
| case | `B=8, H=32, L=2048, D=128, BS=16, MAXB=128, NB=1024` |
| dtype | fp16 |
| q | `[B, H, D]` = 64 KiB |
| k_cache / v_cache | `[NB, BS, H, D]` = 各 128 MiB |
| block_table | `[B, MAXB]` int32 |
| out | `[B, H, D]` = 64 KiB |

## 2. 字节账：这是一个纯带宽问题

真正必须从显存读的是 **KV cache 256 MiB**（q、out、block_table 都是 KB 级）：

```
流量 = 256 MiB = 268,566,528 B
算力 = 4·B·H·L·D = 8.59 GFLOP
AI   = 8.59e9 / 268.6e6 = 32 FLOPs/Byte
```

Roofline 拐点是 **239.7 FLOPs/Byte**，32 远小于它 ⇒ **memory-bound**，算力完全不是瓶颈。

| 参照 | 带宽 | 理论下限 |
|---|---|---|
| 标称峰值 | 272.0 GB/s | 0.99 ms |
| **实测天花板**（`scale` kernel 打出来的） | **221.4 GB/s** | **1.21 ms** |
| 当前 split-L | 215.1 GB/s | 1.2483 ms |

> 本文所有“占峰值”都按 272 标称算，“占实用”按 221.4 算。**只有后者有意义**——这张卡的标称值从来没有达到过。

## 3. 版本台阶

| # | 版本 | 关键改动 | 流量 | ms | GB/s | 占标称 | 占实用 | 相对上一版 |
|---|---|---|---|---|---|---|---|---|
| 1 | 两遍 softmax（naive） | 先扫一遍求 max/分母，再扫一遍算加权和 | 384 MiB | 2.3644 | 170.3 | 62.6% | 76.9% | — |
| 2 | 单遍 online softmax | 用 `m / denom / acc` 增量更新，KV 只读一遍 | 256 MiB | 1.8897 | 142.1 | 52.2% | 64.2% | **1.25×** |
| 3 | 单链 2 路展开（ILP） | 一次处理 2 个 token，两个点积/两路 load 交错 | 256 MiB | 1.5084 | 178.1 | 65.5% | 80.4% | **1.25×** |
| 4 | **split-L（当前）** | 把 L 切 4 段给 4 个 warp 并行，再 merge | 256 MiB | **1.2483** | **215.1** | 79.1% | **97.2%** | **1.21×** |

- **总加速 1.89×**（2.3644 → 1.2483）
- 对 PyTorch 参考实现（Python 显式循环，8.36 ms）= **6.70×**
- 同层另有参考实现（我写的单遍版）1.9763 ms / 135.9 GB/s，与第 2 行差异来自 warp 归约写法和 host 端同步

## 4. 每一步为什么快

### 4.1 两遍 → 单遍（1.25×）

两遍版必须把 KV 读两遍：第一遍算分数找 max 和分母，第二遍才算加权和。**384 MiB 里多出的 128 MiB 就是 K 的第二遍**。

online softmax 把它消掉了：维护运行最大值 `m`、运行分母 `denom`、运行累加 `acc`，每来一个 token 就

```
m_new = max(m, s)
alpha = exp(m - m_new)        // 旧结果的缩放因子
denom = denom * alpha + p
acc   = acc * alpha + p * v
```

好处不只是少读 128 MiB，还因为**只在寄存器里迭代**，没有第二次全局扫描的启动开销。

### 4.2 单遍 → 2 路展开（1.25×）

单遍版每个 warp 每次迭代只有 **2 次 half load**（K 和 V 各一），而一次 load 的延迟是 400~600 个周期。
warp 里只有 2 个未完成的 load，硬件能重叠的请求太少 → 延迟没藏住，带宽只有 142 GB/s。

2 路展开后每次迭代有 **4 个独立的 load**（K0、V0、K1、V1），且它们之间没有依赖：

```cpp
s0 += q * k0[d];   s1 += q * k1[d];      // 两条点积交错
acc[k] = acc[k]*alpha + p0*v0[d] + p1*v1[d];
```

warp 级的等价写法是 **ILP-4**：4 个请求同时在飞，等第一个回来时其余三个已经排好队。
这一步把带宽从 142 推到 178 GB/s。

### 4.3 单链 → split-L（1.21×）

算一下并行度：`B*H = 256` 个 warp = **32 个 block**，而这张卡有 **24 个 SM**。

```
32 / 24 = 1.33 block/SM  →  每 SM 约 10.7 个 warp
SM 的 warp 上限是 64 →  只用了 17%
```

**不是带宽不够，是没有足够的 warp 去喂带宽。** 2 路展开把每 warp 的 ILP 提到了 4，但 warp 总数没变，一旦某个 warp 在等显存，SM 上就没有别的 warp 可以切。

split-L（= vLLM 的 flash-decoding / 分段归约）：

1. 每个 `(b,h)` 的 2048 次迭代切 4 段，交给 4 个 warp 并行；
2. 每段算完交回自己的 `(m_i, d_i, acc_i)`，**不归一化**；
3. 用一个 merge kernel 按 online softmax 的合并公式汇总：

```
M      = max_i m_i
alpha_i = exp(m_i - M)
d      = Σ d_i · alpha_i
acc[k] = Σ acc_i[k] · alpha_i
out    = acc / d
```

并行度 4 倍：`1024 warp = 128 block` → 5.33 block/SM，正好吃满寄存器允许的 5 个 block（`registers_per_thread=48` → 96 regs 上限不触发）。

**代价**：`partial_acc [4,8,32,128]` fp32 = 0.5 MiB 写 + 0.5 MiB 读 = **1 MiB**，只占总量 0.4%。
**语义关键**：split kernel 里绝不能除以 `denom`，否则每段的归一化基准不同，merge 时无法再校正。

## 5. 为什么停在这里

```
215.1 GB/s ÷ 221.4 GB/s（实测天花板）= 97.2%
```

只剩 **2.5%**。bench 报告的 79.1% 是拿 272 标称做的分母，而这张卡从未达到过 272——所有 kernel 的实测最高就是 221.4。

继续调 block 数、展开路数、`half2` 向量化，最多也只能拿回这 2.5%。
**要更快只有一个办法：少搬字节。**

## 6. 正确性验证

- 参考实现是 PyTorch 的**显式循环**（fp32 累加），不是 SDPA——PagedAttention decode 没有对应的 torch 算子，用 SDPA 会掩盖 `block_table` 映射的错误。
- 已验证：`L=2048`（偶数，主循环跑满）、`L=2049`（奇数，走收尾块）。
- **未验证**：`L < BS`（不足一个块，`t/BS` 与 `t%BS` 的边界）——这是欠账，写 README 时不能吹“全长度覆盖”。
- 收尾块曾经有 4 个 bug（`t/BS`、`t%BS`、`d += 32`、`acc[k]`），在 `kernels/CUDA错题本.md` 的 E-73/E-74 有记录。

## 7. 台阶 5：INT8 KV cache（已实测）

| 版本 | 流量 | ms | 实测带宽 | 占 221.4 天花板 |
|---|---|---|---|---|
| fp16 单链 2 路 | 256.0 MiB | 1.5084 | 178.1 GB/s | 80.4% |
| fp16 split-L | 256.0 MiB | 1.2483 | 215.1 GB/s | 97.2% |
| **INT8 K（台阶1）** | **193.1 MiB** | **1.1377** | **178.0 GB/s** | 80.4% |

口径：bench 打印的百分比用标称 272 GB/s，本表「占 221.4」用实测天花板，两个数不要混用。

做法（只动 K，V/Q/out 仍是 fp16）：

- `k_q` = int8 `[NB,BS,H,D]`（128 → 64 MiB），`k_scale` = fp16 `[NB,BS,H]`（0.75 MiB），per-(token, head) 对称量化，`scale = max|x| / 127`。
- 反量化分两次：元素级 `(float)krow[d]`，以及归约后的 `s *= sc`（等价于把 scale 提到求和号外，每 token 只乘 1 次；放进内层就变成每元素乘 1 次，白吃 128 条乘法指令）。
- 误差：`max_abs = 9.77e-4`（参考值量级 0.64）、`max_rel = 9.5e-3`，远小于 1e-2 容差。per-token scale 让每个 token 都吃满 8 位动态范围，误差主要来自 `round()` 取整。

读数：流量 −24.6%，时间 −8.9%（相对 fp16 单链 2 路），带宽利用率 80.4% 不变 —— **INT8 的收益是「字节数线性兑现」**：在 D=128、每 lane 4 次 1 字节加载的模式下扇区利用率仍是 100%，没有付额外指令代价。（第一次跑出 1.3517 ms / 67.7% 是冷启动噪声，不作为结论。）

下一步（按收益排序）：

1. **V 也量化**：流量 193.1 → 137.6 MiB，同利用率下 ≈ 0.81 ms。V 的 scale 可以并进 `p`（`pc = p * sc_v`），每元素不增加指令。
2. **int8×4 向量化加载**：每 lane 读 4 字节，L1 事务数降到 1/4，目标把利用率从 80% 推到 90%+。
3. **dp4a**：Q 也量化成 int8，一条指令做 4 次乘加——此时才真正用上整数算力。

## 8. 文件

| 文件 | 是什么 |
|---|---|
| `kernels/w7_paged_attn_split.cu` | **当前最优**（split-L，1.2483 ms） |
| `kernels/w7_paged_attn.cu` | 单链 2 路展开（1.5084 ms） |
| `kernels/w7_paged_attn_单遍online.cu.bak` | 单遍 online（1.8897 ms） |
| `kernels/w7_paged_attn_参考版.cu.bak` | 参考实现（1.9763 ms） |
| `kernels/w8_int8_attn.cu` | INT8 K（台阶1，1.1377 ms） |
| `kernels/w8_INT8台阶1-实现契约.md` | 台阶1 的契约与 19 处改动清单 |
| `kernels/bench_kernels.py` | REGISTRY 里 `paged_attn` / `paged_attn_split` / `paged_attn_int8` 三条 |

## 9. 面试能讲的三点

1. **为什么 decode 是带宽题**：AI=32 vs 拐点 240，算力闲着，瓶颈是 256 MiB 的 KV 读。
2. **为什么 2 路展开有用而 4 路没继续**：ILP 补的是“单 warp 内请求数”，split-L 补的是“warp 总数”；前者天花板是寄存器，后者天花板是 SM 数量。到 97% 实测带宽后两者都没意义了。
3. **split-L 的坑在归一化时机**：每段用各自的 `m_i` 算出的 `d_i` 和 `acc_i` 是不同基准的，必须在 merge 时用 `exp(m_i - M)` 校正；提前归一化就无法还原。
4. **INT8 省的是搬运不是算力**：decode 阶段算力闲置，int8 的全部价值是把 K 的 128 MiB 砍到 64 MiB；实现上仍是「读 int8 → 转 float → fp32 累加」，真正的整数算力要等 dp4a。scale 必须乘在归约之后（`Σ q·(sc·k) = sc·Σ q·k`），否则每元素多一条乘法。
