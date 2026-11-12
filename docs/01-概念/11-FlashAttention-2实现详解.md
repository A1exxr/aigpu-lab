# FlashAttention-2 实现详解：从论文的三条改动到可运行的 kernel

> 这份文档和另外两篇的分工：
> - `04-attention入门.md`：naive attention 的流量账（为什么要 FA）
> - `05-flash-attention详解.md`：online softmax 的数学推导（FA **在算什么**）
> - `10-flash-attention反向.md`：反向的五个公式（**梯度怎么来**）
> - **本文**：FA2 相对 FA1 改了什么、每条改动值多少、以及**具体怎么写出来**（tile 尺寸、smem、寄存器、warp 划分、causal、GQA、反向双 kernel）
>
> 目标产物：`kernels/w12_fa2.py`（Triton，本地 4060 可完整跑通），并在面试里能讲清「v1 和 v2 的差别在哪一行代码」。

---

## 0. 先记住一句话

**FA1 解决了「显存放不下」，FA2 解决了「放了但算得慢」。**

FA1 把 attention 从「必须物化 S×S 矩阵」变成「分块 + online softmax，显存线性」，但它在 A100 上只能跑到理论峰值的 **25–40%**。
FA2 不动数学，只改**工作划分**（work partitioning），把利用率推到 **50–73%**，端到端训练到 **225 TFLOPs/s（72% MFU）**，比 FA1 快约 **2×**（[arXiv:2307.08691](https://arxiv.org/abs/2307.08691)）。

原文三条改动（摘要原话的翻译）：

| # | 论文原话 | 落到代码上是 |
|---|---|---|
| 1 | tweak the algorithm to reduce the number of non-matmul FLOPs | 少做缩放/最大值/求和的标量运算，`exp` 换 `exp2`，除以 `l` 推到循环外 |
| 2 | parallelize the attention computation, even for a single head, across different thread blocks | **grid 换成按 Q 分块**，O 留在寄存器，长序列时一个 head 也能占满 SM |
| 3 | within each thread block, distribute the work between warps to reduce communication through shared memory | **4 个 warp 沿 Q 的行切分**，S/O 都是 warp 私有，只有 K/V 共享 |

**第 2 条是主因，第 3 条是次因，第 1 条只值几个百分点。** 后面会算给你看。

---

## 1. FA1 到底慢在哪：O 被反复读写

### 1.1 两个版本的循环顺序

```
FA1（Algorithm 1）                    FA2（Algorithm 2）
for j in KV 块:           ← 外层      for i in Q 块:            ← 外层
    for i in Q 块:        ← 内层          for j in KV 块:       ← 内层
        S = Q_i K_j^T                         S = Q_i K_j^T
        online softmax 更新 m, l              online softmax 更新 m, l
        O_i = diag(alpha) O_i + P V_j         O_i = diag(alpha) O_i + P V_j
        ↑ O_i 从 HBM 读、改、写回              ↑ O_i 一直在寄存器

grid = (n_kv_blocks, B*H)              grid = (n_q_blocks, B*H)
```

### 1.2 流量差多少（S=2048, D=128, H=32, fp16, Br=Bc=64）

| 项 | FA1 | FA2 |
|---|---|---|
| O 的 HBM 往返 | 每个 KV 块都要读+写一遍 O：`n_kv × S × D × 2 B × 2(读写)` | 只在最后写一次：`S × D × 2 B` |
| 单 head 数字 | 32 × 2048 × 128 × 2 × 2 = **32 MiB** | 2048 × 128 × 2 = **0.5 MiB** |
| × 32 heads | **1 GiB** | **16 MiB** |

**64 倍的 O 流量差。** 这不是理论推导——`ncu` 里看 `dram__bytes_write` 就能直接验证。

> 注意：K/V 的读放大两个版本一样（每个 Q 块都要把 K/V 全读一遍，共 `S/Br` 次）。FA2 没有也不可能做到「K/V 只读一次」，这是 `05` 文档 §11 的第 3 条误区。

### 1.3 并行度差多少

| 场景 | FA1 的 grid | FA2 的 grid |
|---|---|---|
| S=2048, B=1, H=32 | `n_kv=32` × 32 = 1024 blocks | `n_q=32` × 32 = 1024 blocks |
| **S=128（短序列）** | `n_kv=2` × 32 = 64 blocks | `n_q=2` × 32 = 64 blocks |
| **B=1, H=8, S=4096** | 64 × 8 = 512 | 64 × 8 = 512 |

看起来一样？**差别在「一个 block 里干了多少活」**：

- FA1 的每个 block 必须**扫完整个 Q**（`n_q` 次内层迭代），O 在 HBM 上被改 `n_q` 次；
- FA2 的每个 block 只负责 **Br 行 Q**，内层扫 KV，O 全程在寄存器。

所以 FA2 的 block 更「短命」、更容易被调度填满，且没有 O 的 HBM 依赖链。**短序列（S < 2×Br）时 FA2 仍然并行度不足**——这时要靠 **split-KV**（把 KV 再切一段，各算 partial，最后 merge）。这个你在 `w7_paged_attn_split.cu` 里已经做过（`1.5084 → 1.2483 ms`），是同一个套路。

---

## 2. 改动一：非 matmul 的 FLOPs（为什么只值几个百分点）

### 2.1 一个 (i, j) 块内的 FLOPs 账

设 Q 块 `Br × D`、KV 块 `Bc × D`：

| 操作 | 次数 | 类型 |
|---|---|---|
| `S = Q K^T` | `2 · Br · Bc · D` | matmul（Tensor Core） |
| `O += P V` | `2 · Br · Bc · D` | matmul（Tensor Core） |
| `exp(S - m)` | `Br · Bc` 次 exp ≈ `4 · Br · Bc` FLOPs | **非 matmul**（SFU） |
| 行 max / 行 sum | `2 · Br · Bc` | 非 matmul |
| `O *= alpha` | `Br · D` | 非 matmul |
| `l = l·alpha + rowsum` | `Br` | 非 matmul |

matmul 合计 `4 Br Bc D`，非 matmul 约 `6 Br Bc + Br D`。

比例：

```
非 matmul / matmul = 6/(4D) + 1/(4Bc)
D=128, Bc=64 → 6/512 + 1/256 = 1.17% + 0.39% ≈ 1.6%
```

**所以论文第 1 条改动最多省 1–2% 的时间**（再算上 SFU 和 Tensor Core 并行、exp 可以藏在 matmul 后面，实际更低）。

那为什么还要做？因为**免费**：

```python
# 三招，一行不改数学
scale = sm_scale * 1.44269504          # log2(e)，把 exp 换成 exp2
p = tl.math.exp2(s * scale - m_new * scale)
o = acc / l[:, None]                   # 只在最后除一次，不在循环里除
```

`exp2` 直接映射到 `MUFU.EX2` 一条指令；`exp` 要先乘 `log2(e)`。**在 SASS 里能看到这个差别**（`cuobjdump -sass | grep MUFU`）。

### 2.2 面试怎么说

> 「FA2 论文第一条是减少非 matmul FLOPs，我算过：`exp`/`max`/`sum` 加起来约是 matmul 的 1.6%（D=128, Bc=64），所以这条只值 1–2%。FA2 真正的 2× 来自并行度和 warp 分工。论文里那条主要是因为它免费。」

这话比背「三个改进」有用得多——说明你算过。

---

## 3. 改动三：warp 怎么分工（比改动二更容易写错）

### 3.1 错误切法：沿 D 切

```
warp0 算 D[0:32]，warp1 算 D[32:64]，...
```

问题：`S = Q K^T` 是**沿 D 求和**的，每个 warp 只算了一部分和，必须跨 warp 归约 → 要么走 smem + `__syncthreads`，要么 `__shfl` 跨 warp（做不到）。而且行 max/sum 也变成跨 warp。

**这就是 FA1 慢的原因之一：块内通信多、barrier 多。**

### 3.2 正确切法：沿 Br（Q 的行）切

```
Br = 64, 4 warps → 每个 warp 管 16 行 Q
warp w 负责 Q 的第 [w*16, w*16+16) 行

S 的这 16 行 = Q_rows(16×D) @ K^T(D×Bc)   ← 只用自己那 16 行 Q
行 max / 行 sum / P / O 的 16 行          ← 全是 warp 私有
只有 K/V 块是 4 个 warp 共享的（放 smem）
```

| 量 | 归属 | 需要跨 warp 吗 |
|---|---|---|
| `Q` 的 16 行 | warp 私有（寄存器） | ❌ |
| `S` (16×Bc)、`m`、`l` | warp 私有 | ❌ |
| `O` 的 16 行 (16×D) | warp 私有（寄存器） | ❌ |
| `K` 块、`V` 块 | **共享 smem** | 只在装载后需要 1 次 `__syncthreads` |

**结论：一个 warp 内部的 S→softmax→PV 全程不需要任何 barrier。** 每装载一个新的 K/V 块，才需要一次 `__syncthreads`（保证 4 个 warp 都看到数据）。

这就是论文第 3 条「distribute work between warps to reduce communication through shared memory」的具体含义。

### 3.3 伪代码（前向，causal + GQA）

```python
@triton.jit
def _fa2_fwd(Q, K, V, O, L, M,       # L=lse, M 可省（只调试用）
             stride_qz, stride_qh, stride_qm, stride_qd,
             stride_kz, stride_kh, stride_kn, stride_kd,
             stride_vz, stride_vh, stride_vn, stride_vd,
             stride_oz, stride_oh, stride_om, stride_od,
             Z, H, HKV, N_CTX, sm_scale,
             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr):
    start_m = tl.program_id(0)          # Q 块编号
    off_hz  = tl.program_id(1)          # (batch, head) 打平
    off_z   = off_hz // H
    off_h   = off_hz % H
    off_hkv = off_h // (H // HKV)       # ★ GQA：q head → kv head 的映射

    q = tl.load(Q + off_z*stride_qz + off_h*stride_qh
                + (start_m*BLOCK_M + tl.arange(0, BLOCK_M))[:, None] * stride_qm
                + tl.arange(0, HEAD_DIM)[None, :] * stride_qd)     # [BM, D]

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)

    # ★ causal：只扫到对角块为止（多扫的块白算）
    hi = tl.minimum((start_m + 1) * BLOCK_M, N_CTX) if IS_CAUSAL else N_CTX
    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k = tl.load(K + ... , mask=offs_n[:, None] < N_CTX)        # [BN, D]
        v = tl.load(V + ... , mask=offs_n[:, None] < N_CTX)        # [BN, D]

        s = tl.dot(q, tl.trans(k)) * sm_scale                      # [BM, BN]
        if IS_CAUSAL:                                              # 对角块要打 mask
            s = tl.where(offs_m[:, None] >= offs_n[None, :], s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.math.exp2((m_i - m_new) * 1.44269504)
        p = tl.math.exp2(s * 1.44269504 - m_new[:, None] * 1.44269504)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)    # ★ 一行同时 rescale + 累加
        m_i = m_new

    acc = acc / l_i[:, None]
    lse = m_i + tl.math.log2(l_i) * 0.6931471805599453               # 反向要用
    tl.store(O + ..., acc.to(O.dtype.element_ty))
    tl.store(L + ..., lse)
```

**注意 `acc = acc * alpha[:, None] + tl.dot(p, v)` 这一行**：`acc*alpha` 是 `BM×D` 次乘（非 matmul），`tl.dot` 是 matmul，Triton 会把它们编成一条 `mma` + 一条 `mul`，可能重叠执行。FA1 这里要把 `acc` 写回 HBM 再读回来，FA2 全程寄存器。

---

## 4. 实现蓝图：本机 4060 上的具体配置

目标 shape（Llama-3 8B 的 attention）：`B=1, H=32, HKV=8, S=2048, D=128, fp16, causal`。

### 4.1 三组候选配置的账（估算，必须用 ncu 验证）

| 配置 | Br / Bc | 线程 | acc 寄存器/线程 | S 寄存器/线程 | smem/级 | 3 级 smem |
|---|---|---|---|---|---|---|
| A | 64 / 64 | 128（4 warp） | `64×128×4 / 128 = 32` | `64×64×4/128 = 16` | K 16 KB + V 16 KB = 32 KB | 96 KB ❌ 超 100 KB 上限 |
| B | 64 / 32 | 128（4 warp） | 32 | 8 | K 8 + V 8 = 16 KB | 48 KB ✅ |
| C | 128 / 64 | 256（8 warp） | `128×128×4/256 = 32` | `128×64×4/256 = 16` | 32 KB | 64 KB（2 级）✅ |

4060 单 SM：**寄存器 64K**（每线程上限 255）、**smem 100 KB**（opt-in）。取 B 或 C：

- **B（Br=64, Bc=32, 128 线程）**：每 block 寄存器约 `128 × (32+16+32 杂项) ≈ 10K` → 可驻留 2–4 个 block，占用率好；缺点是 `Bc=32` 使 `exp` 的标量开销占比上升（1.6% → 2.3%）。
- **C（Br=128, Bc=64, 256 线程）**：Tensor Core 效率更高（`m16n8k16` 切得更满），但 1 个 block/SM 时只有 8 warp，延迟隐藏要靠 ILP。

**建议：先写 B（好调试、占用率高），跑通后改 C 比数字。** 这和当初 GEMM 从 32×32 到 64×64 的顺序一样。

### 4.2 smem 里放什么

| 数据 | 放哪 | 为什么 |
|---|---|---|
| `Q` 块 | **不放 smem**（每 warp 自己那 16 行 → 寄存器） | 沿 M 切分后 Q 不共享 |
| `K` 块 | smem，`Bc × D` | 4 个 warp 都要读全部 K |
| `V` 块 | smem，`Bc × D` | 同上 |
| `S` / `P` | 寄存器（`16×Bc` per warp） | warp 私有 |
| `acc` | 寄存器（`16×D` per warp） | warp 私有 |

`K` 在 smem 里会被 4 个 warp 同时读**同一行**（broadcast，不冲突）；每个 warp 读的是不同行的 `Q`（在寄存器里，不涉及 smem）。

### 4.3 反过来的写法（也有实现这么做）

沿 `Bc`（KV 的列）切 warp：每个 warp 算 S 的一段列，行 max/sum 要跨 warp 归约 → 需要 smem + barrier。**FA2 论文没这么干**，但 split-KV 外层再切一刀时会变成这样。写的时候别混。

---

## 5. causal 的两种实现，别只做一种

| 层次 | 做法 | 省什么 |
|---|---|---|
| **块级跳过** | `for start_n in range(0, min((start_m+1)*Br, S), Bc)` | 省掉对角块**以上**的全部计算（约一半） |
| **块内 mask** | `s = tl.where(offs_m[:, None] >= offs_n[None, :], s, -inf)` | 对角块内部只算一半，但**计算照做** |

只做块内 mask 不做块级跳过，等于白算一半的块——这是 FA2 实现里最常见的性能损失。判据：`ncu` 里 `sm__pipe_tensor_cycles_active` 应接近非 causal 版本的一半左右（不是 1/4，因为对角块仍有浪费）。

---

## 6. GQA：一行索引，但要三处一起改

Llama-3 8B：`H=32` 个 q head，`HKV=8` 个 kv head，`group = H/HKV = 4`。

```python
off_hkv = off_h // (H // HKV)     # 前向：q head → kv head
```

三处都要用 `off_hkv` 而不是 `off_h`：

1. K/V 的 base pointer（`stride_kh` / `stride_vh`）
2. 反向里 dK/dV 的累加目标（**4 个 q head 会往同一个 kv head 累加 → 要么 atomic，要么在 dkdv kernel 里对 4 个 q head 求和**）
3. K/V 的 cache 形状（`[B, HKV, S, D]`，不是 `[B, H, S, D]`）

**GQA 反向是 v2 实现里最容易错的地方**：dK/dV 的累加维度从「1 个 head」变成「group 个 head」，如果你按 `[B,H,...]` 分配 dK，会多分配 4 倍并且 3/4 是垃圾。

---

## 7. 反向：从「能算」到「确定、不靠 atomic」

### 7.1 五个公式（和 `10-flash-attention反向.md` 一致）

```
delta = rowsum(dO ∘ O)                    # [B,H,S]，一次 elementwise reduce
dp    = dO @ V^T
ds    = P ∘ (dp - delta[..., None])
dv    = P^T @ dO
dk    = ds^T @ Q * scale
dq    = ds   @ K * scale
```

`P` 不存，用保存的 `lse` 重算：`p = exp(s - lse)`。

### 7.2 FA2 官方实现为什么是**两个 kernel**

| kernel | 外层循环 | 累加谁 | 为什么这样切 |
|---|---|---|---|
| **dkdv** | KV 块 `j` | `dK_j, dV_j` 在寄存器 | dK/dV 只依赖「所有 i 对这一块 j 的贡献」→ 一个 block 负责一个 j，内层扫 i，**无冲突** |
| **dq** | Q 块 `i` | `dQ_i` 在寄存器 | dQ 只依赖「i 这一行对所有 j」→ 一个 block 负责一个 i，内层扫 j |

两个 kernel 各自都没有跨 block 的写冲突 ⇒ **不需要 atomicAdd，结果逐位可复现**。

对比你 `w11_bwd_3_split.py` 的做法（写 partial + merge）：本质相同，都是「把冲突的累加变成确定性的两趟」。区别是官方用「切循环」而不是「切输出」。

### 7.3 反向的 smem / 寄存器（dkdv kernel，Br=64, Bc=64, D=128）

| 数据 | 位置 | 大小 |
|---|---|---|
| `K_j, V_j` | 寄存器 + smem | `Bc × D` each |
| `dK_j, dV_j` | **寄存器累加器** | `Bc × D` each（fp32）→ 每线程 `2 × 64×128×4 / 128 = 128` 个 |
| `Q_i, dO_i, O_i, lse_i, delta_i` | 每轮从 HBM 读 | `Br × D` / 标量 |
| `S, P, dP, dS` | 寄存器（临时） | `Br × Bc` per warp |

**注意 `dK/dV` 的累加器占了 128 个寄存器/线程**——这是反向比前向难写的核心原因（前向 acc 只 32 个）。Bc 取 32 可以减半，代价是内层循环次数翻倍。

---

## 8. 我们现在到哪了：与 `w5` / `w11` 的逐项差距

| 能力 | FA1（`w5_flash_attn.py`） | FA2 要求 | 差距 |
|---|---|---|---|
| online softmax | ✅ | ✅ | — |
| Q 外层 + O 留寄存器 | ❌（Triton 默认写法是 K/V 外层） | ✅ | 改 grid 语义 |
| warp 沿 M 切分 | Triton 自动（`num_warps`） | 手动控制 | Triton 里只能间接影响 |
| causal（块级跳过 + mask） | ❌ | ✅ | **必须加** |
| GQA | ❌ | ✅ | **必须加** |
| 反向 | `w11`（Triton，split + atomic） | 无 atomic、确定性、两 kernel | 改累加策略 |
| 性能口径 | 0.474 ms / 18.1 TFLOPS / SDPA 91% | 目标 ≥95% SDPA | 需重测 |

> `w5` 现在算 **FA1 + 部分 v2 思想**（Q 外层写了，但没 causal、没 GQA、没有针对性 warp 分工）。**不能叫 v2**，面试里说「实现了 FA1 的 online softmax 版本，正在做 v2 的 causal/GQA 和反向」。

---

## 9. 验收判据（写死，别跑完再想）

| # | 判据 | 怎么测 |
|---|---|---|
| 1 | 前向 vs SDPA（causal, fp16） | `max_abs_diff < 2e-3` |
| 2 | 反向 vs autograd | `max_rel < 2e-2`（fp16）；`torch.equal` 同输入两次（确定性） |
| 3 | GQA 三种配置 | `(H,HKV) ∈ {(32,32),(32,8),(32,1)}` 都能跑且对 |
| 4 | causal 正确性 | 与 `F.scaled_dot_product_attention(is_causal=True)` 比；另测 `S=1, S=17, S=2048` 三种长度 |
| 5 | 性能 | S=2048 causal fp16，≥ SDPA 的 95%；报 TFLOPS 与 SM 利用率 |
| 6 | 与 v1 对比 | 同 shape 下 v2 / v1 的毫秒比（预期 1.5–2.5×） |

**第 5 条的口径**：`TFLOPS = 4 · S² · D · H · 0.5(causal) / 时间`，与 SDPA 用同一个公式算，否则不可比。

---

## 10. 十二个高频 bug（按发生概率排）

| # | 症状 | 原因 |
|---|---|---|
| 1 | 结果偏大/偏小一个常数 | `scale` 用两次（`q*scale` 后 `s*scale`）或 `1/sqrt(D)` 挂在错的位置 |
| 2 | 只有第一个 token 对 | `lse` 存的是 `m` 不是 `m + log(l)` |
| 3 | 对角线附近错 | 块内 mask 的 `>=` 写成 `>`；或 `offs_m`/`offs_n` 反了 |
| 4 | 长序列 NaN | `m_i` 初值写成 0（不是 `-inf`）；或 `p` 在整行 mask 后全 0 时 `l=0` |
| 5 | 反向 dK/dV 差 4 倍 | GQA 里 dK 按 `[B,H,...]` 分配，实际应累加到 `[B,HKV,...]` |
| 6 | 反向 dQ 对、dK 错 | dkdv kernel 里 `ds^T @ q` 用了 `ds @ q^T` |
| 7 | 两次跑结果不同 | 用了 `atomic_add`（fp16 原子加顺序不定） |
| 8 | 短序列慢 | 没做 split-KV，grid 只有 `ceil(S/Br)` 个块 |
| 9 | causal 只快 10% | 只做了块内 mask，没做块级跳过 |
| 10 | fp16 溢出成 inf | `acc` 用 fp16 累加（必须 fp32） |
| 11 | 和 SDPA 差 1e-2 | SDPA 默认可能用 TF32/不同 backend；对比时显式指定 `backend` 或关掉 |
| 12 | Triton 编译 OOM | `num_stages` 太高（每级都要 smem），降到 2 |

---

## 11. 面试口径（背这 6 句）

1. **FA1 vs FA2**：「FA1 用 online softmax 把显存从 O(S²) 降到 O(S)；FA2 不改数学，改工作划分——Q 外层让 O 全程在寄存器、grid 按 Q 块切、块内沿 Q 的行切 warp，把 A100 利用率从 25–40% 提到 50–73%。」
2. **为什么 Q 外层更好**：「FA1 每个 KV 块都要把 O 读改写一遍 HBM，S=2048、D=128、H=32 时是 1 GiB 的 O 流量；FA2 只写一次，16 MiB。」
3. **warp 怎么切**：「沿 Q 的行切，S/O/softmax 全 warp 私有，K/V 共享 smem，只有装 K/V 时需要 barrier。沿 D 切会逼出跨 warp 归约。」
4. **causal 怎么省**：「外层循环上限改成对角块，块内再 mask；只做 mask 不做跳过，等于白算一半。」
5. **GQA**：「`kv_head = q_head // group`，K/V cache 形状是 `[B, HKV, S, D]`；反向里 dK/dV 要跨 group 个 q head 累加，这是最容易错的地方。」
6. **反向为什么两个 kernel**：「dkdv 和 dq 各切一个循环维度，各自没有写冲突，所以不用 atomic、结果逐位可复现；用 atomic 的版本每次跑都不一样。」

---

## 12. 参考文献

- Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning", [arXiv:2307.08691](https://arxiv.org/abs/2307.08691)（摘要里的三条改动 + 50–73% / 225 TFLOPs/s 数字都出自这里）
- Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
- 官方实现：[github.com/Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)（`csrc/flash_attn/src/flash_fwd_kernel.h` 看 warp 划分，`flash_bwd_kernel.h` 看两个 kernel 的循环）
- Triton 官方 tutorial `06-fused-attention.py`（最接近 FA2 前向的可读实现）
- 本仓库：`05-flash-attention详解.md`（推导）、`10-flash-attention反向.md`（梯度公式）、`kernels/w5_flash_attn.py`（现有 v1 实现）
