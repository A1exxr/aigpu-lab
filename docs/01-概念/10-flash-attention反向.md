# FlashAttention 反向：从公式到 Triton

> 前置：`05-flash-attention详解.md`（online softmax）、`w5_flash_attn.py`（你自己的前向，0.474 ms）。
> 目标：能写出 $dQ/dK/dV$ 并与 `torch.autograd` 对上；知道为什么反向要"重算"而不是"读回"。
> 规模按笔记本能跑的来：$B=1,\ H=8,\ S=1024,\ d=64$。

> 公式用 LaTeX 写（行内 `$...$`、独立 `$$...$$`）：VS Code 预览、Typora、Obsidian、GitHub 都能直接渲染。

---

## 0. 一句话

**反向 = 用前向保存的 $O, m, \ell$ 和上游的 $dO$，在分块循环里重算一遍 $P$，从而在不物化 $S\times S$ 的前提下得到 $dQ, dK, dV$。**

---

## 1. 前向存了什么，为什么不能存 $S$

前向：

$$
S=\frac{QK^\top}{\sqrt d},\qquad P=\operatorname{softmax}(S)\ \text{(按行)},\qquad O=PV
$$

反向需要 $P$；而 $P=\operatorname{softmax}(S)$ 需要 $S$ 本身，$S$ 是 $S\times S$ 的：

$$
S=4096,\ d=128:\quad 4096^2\times 2\ \text{B}=32\ \text{MiB（fp16，单头）}
$$

所以前向**只保存两个 $S\times d$ 的小量**（每行一个标量）：

$$
m_i=\max_j S_{ij},\qquad \ell_i=\sum_j e^{S_{ij}-m_i}
$$

反向现场重算：

$$
P_{ij}=\frac{e^{S_{ij}-m_i}}{\ell_i},\qquad S_{ij}=\frac{Q_i\cdot K_j}{\sqrt d}
$$

代价是多花 $2S^2d$ FLOPs 重算 $S$，换来省掉 $O(S^2)$ 显存。

---

## 2. 五个梯度公式

已知上游梯度 $dO\in\mathbb{R}^{S\times d}$，以及 $O,P,Q,K,V$（$P$ 可现场重算）：

$$
\begin{aligned}
dV &= P^\top dO &&\in\mathbb{R}^{S\times d}\\
dP &= dO\,V^\top &&\in\mathbb{R}^{S\times S}\\
\delta_i &= \sum_j P_{ij}\,dP_{ij} &&\in\mathbb{R}^{S}\ \text{(每行一个标量)}\\
dS &= P\odot\big(dP-\delta\,\mathbf{1}^\top\big) &&\in\mathbb{R}^{S\times S}\\
dQ &= \frac{dS\,K}{\sqrt d},\qquad dK=\frac{dS^\top Q}{\sqrt d} &&\in\mathbb{R}^{S\times d}
\end{aligned}
$$

$\odot$ 是逐元素乘；$\delta\,\mathbf{1}^\top$ 表示把列向量 $\delta$ 按行广播成 $S\times S$（第 $i$ 行全是 $\delta_i$）。

> $dS$ 里的 $-\delta$ 来自 softmax 的 Jacobian：
> $$
> \frac{\partial P_{ij}}{\partial S_{ik}}=P_{ij}\big(\mathbf{1}[j=k]-P_{ik}\big)
> $$
> 对 $k$ 求和后，第一项给出 $dP$、第二项给出 $-\delta$。这就是"$dS$ 行和为 0"的来源。

### 2.1 $\delta$ 是什么，为什么能写成 $\operatorname{rowsum}(dO\odot O)$

从定义出发，把 $dP=dO\,V^\top$ 代进去：

$$
\begin{aligned}
\delta_i
&=\sum_j P_{ij}\,dP_{ij}
=\sum_j P_{ij}\,\big(dO_i\cdot V_j\big)\\
&=dO_i\cdot\Big(\sum_j P_{ij}V_j\Big)
=dO_i\cdot O_i
\end{aligned}
$$

所以

$$
\boxed{\ \delta=\operatorname{rowsum}(dO\odot O)\ }
$$

复杂度 $O(Sd)$ 而不是 $O(S^2)$——**反向不必等 $dP$ 算完就能先拿到 $\delta$**，这是分块实现能成立的关键。

> **坑 1**：$\delta$ 是 $\operatorname{rowsum}(dO\odot O)$，不是 $\operatorname{rowsum}(dO\odot dO)$。
> **坑 2**：$-\delta$ 必须**按行广播**减到每个 $dP_{ij}$ 上；漏了它 $dS$ 的行和不为 0，结果"看起来差不多"但全错。

### 2.2 $1/\sqrt d$ 挂在哪

$$
S=\frac{QK^\top}{\sqrt d}\ \Longrightarrow\ \frac{\partial S}{\partial Q}=\frac{K}{\sqrt d},\qquad
dQ=\frac{dS\,K}{\sqrt d},\quad dK=\frac{dS^\top Q}{\sqrt d}
$$

$dS$ 是**对 $S$ 的梯度**，缩放因子在把 $dS$ 乘回 $Q,K$ 时才除；$dS$ 自身不再除一次。

---

## 3. 分块反向：谁累加谁

不分块时 $dP,dS$ 都要物化 $S\times S$（$S=4096$ 单头 fp32 就是 $64$ MiB），必爆。所以要分块。

### 3.1 循环结构（FA2 / Triton tutorial 的做法）

```
外层：遍历 Q 块 i（块大小 BM）
    dq_i 的累加器放寄存器（BM × d）
    内层：遍历 K/V 块 j（块大小 BN）
        s_ij  = q_i k_j^T / sqrt(d)          ← 重算
        p_ij  = exp(s_ij - m_i) / l_i        ← 用前向保存的 m_i, l_i
        dp_ij = do_i v_j^T
        ds_ij = p_ij ∘ (dp_ij - delta_i)
        dq_i += ds_ij k_j / sqrt(d)          ← 只依赖 i → 就地累加
        dk_j += ds_ij^T q_i / sqrt(d)        ← 依赖所有 i → 冲突
        dv_j += p_ij^T do_i                  ← 依赖所有 i → 冲突
```

| 梯度 | 依赖范围 | 能否就地累加 |
|---|---|---|
| $dQ_i$ | 只依赖块 $i$（内层对 $j$ 求和） | ✅ 寄存器累加 |
| $dK_j$ | 需对所有 $i$ 求和 | ❌ 跨 block |
| $dV_j$ | 需对所有 $i$ 求和 | ❌ 跨 block |

**为什么 $P_{ij}$ 必须用全局的 $m_i,\ell_i$？** 因为 $\ell_i=\sum_j e^{S_{ij}-m_i}$ 是**整行**的和；只用块内 max/sum 得到的是"块内 softmax"，乘出来的 $dQ$ 缺项、$dK/dV$ 偏大，且 $S$ 越大误差越大。FA2 相对 FA1 的改进之一就是反向直接用保存值、不做 online rescale。

### 3.2 两种 $dK/dV$ 累加策略

| 策略 | 做法 | 优点 | 缺点 |
|---|---|---|---|
| **A. atomicAdd** | `tl.atomic_add(dK_ptr, dk_block)` | 代码最短（Triton 官方 tutorial 就这么写） | fp16 原子加精度差、结果**非确定** |
| **B. split + 归约** | 开 $[n_q, S, d]$ 的 fp32 缓冲，每块写自己那份，最后一趟求和 | 确定、精度好 | 多一份显存 $n_q\,S\,d\times 4$ B，多一趟 reduce |

建议：**先写 A 把公式跑对，再写 B 出可复现数字**。你在 PagedAttention split-L 上做过的"分块写 partial + 最后 merge"是同一套路。

---

## 4. Triton 实现骨架

```python
@triton.jit
def _attn_bwd(Q, K, V, sm_scale, O, dO, dQ, dK, dV,
              M, L,                        # 前向保存的 m、l
              stride_qz, stride_qh, ...,
              Z, H, N_CTX, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr):
    start_m = tl.program_id(0)
    off_hz  = tl.program_id(1)

    # ① 载入 q_i, do_i, o_i, m_i, l_i
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    q  = tl.load(Q + ...)          # (BM, D)
    do = tl.load(dO + ...)         # (BM, D)
    o  = tl.load(O + ...)          # (BM, D)
    m  = tl.load(M + offs_m)       # (BM,)
    l  = tl.load(L + offs_m)       # (BM,)

    dq    = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    delta = tl.sum(do.to(tl.float32) * o.to(tl.float32), axis=1)   # delta = rowsum(dO ∘ O)

    # ② 内层遍历 K/V 块
    for start_n in range(0, N_CTX, BLOCK_N):
        k  = tl.load(K + ...)                      # (BN, D)
        v  = tl.load(V + ...)                      # (BN, D)
        s  = tl.dot(q, tl.trans(k)) * sm_scale     # (BM, BN) 重算
        p  = tl.exp(s - m[:, None]) / l[:, None]   # 用全局 m、l
        dp = tl.dot(do, tl.trans(v))               # (BM, BN)
        ds = p * (dp - delta[:, None])             # Jacobian 项
        dq += tl.dot(ds.to(q.dtype), k) * sm_scale
        dk  = tl.dot(tl.trans(ds).to(k.dtype), q) * sm_scale
        dv  = tl.dot(tl.trans(p).to(v.dtype), do)
        tl.atomic_add(dK + ..., dk)                # 策略 A
        tl.atomic_add(dV + ..., dv)

    tl.store(dQ + ..., dq.to(Q.dtype.element_ty))
```

**四个 Triton 特有的坑**：

1. `tl.dot` 的两个操作数必须是 fp16/bf16，所以 `ds.to(q.dtype)`、`p.to(v.dtype)` 不能省；**转换放在归约之后**（累加仍用 fp32）。
2. `tl.trans` 只改 stride（逻辑转置），不搬数据，所以 `tl.dot(tl.trans(ds), q)` 不会多一次访存。
3. `atomic_add` 的偏移是 $[\text{off\_hz},\ \text{offs\_n},\ \text{offs\_d}]$（$dK$ 沿 key 维），写成 `offs_m` 会把梯度写到错误的行。
4. 广播要手动：`m / l / delta` 是 `(BM,)`，必须写 `[:, None]` 才能和 `(BM, BN)` 相减。

---

## 5. 验收标准（写死在文件顶部，别改）

| 步骤 | 验收 |
|---|---|
| PyTorch 手写版 | $\max\lvert dQ-dQ_{\text{ref}}\rvert < 10^{-5}$（fp32） |
| Triton 单块版（$S=128$） | $\max\lvert\cdot\rvert < 2\times10^{-3}$（fp16 输入、fp32 累加） |
| Triton 分块版（$S=1024$） | 同上，且三次跑差异 $<2\%$（否则改用策略 B） |
| 性能 | 反向时间 $\le 3\times$ 前向（前向 $0.474$ ms → $\le 1.5$ ms） |

**FLOPs 预算**（每头）：

$$
\text{forward}=4S^2d,\qquad
\text{backward}=\underbrace{2S^2d}_{\text{重算 }S}+\underbrace{8S^2d}_{dV,\,dP,\,dQ,\,dK}=10S^2d=2.5\times\text{forward}
$$

所以"反向比前向慢 2.5 倍"是正常的；慢到 5 倍以上说明实现有问题（重复重算或没复用 $S$）。

---

## 6. 六个高频 bug（按发生概率排）

1. **$\delta$ 算错**：写成 $\operatorname{rowsum}(dO\odot dO)$ 或干脆没算。现象：$dQ$ 整体差一个常数倍。
2. **$P$ 用了块内 $\ell$**：$dQ$ 缺项、$dK/dV$ 偏大。现象：$S$ 越大误差越大。
3. **$-\delta$ 没按行广播**：$dS$ 行和不为 0。**验证：打印 `ds.sum(axis=1)`，应全为 0（$10^{-6}$ 内）**——最好用的自检。
4. **$1/\sqrt d$ 用错位置**：$dS$ 又除一次，结果整体小 $1/\sqrt d$。
5. **$dK/dV$ 偏移用 `offs_m`**：写到错误行。现象：部分位置对、部分为 0。
6. **fp16 原子加**：尾数丢失。现象：单次跑对、跑两次不同、batch 变大误差变大。

**自检三板斧**：
- `ds.sum(axis=1) ≈ 0`（验 Jacobian 项）
- $dQ_{\text{ref}}-dQ$ 的**行均值** $\approx 0$（验 $\delta$）
- 手算小矩阵（$S=4,\ d=2$）逐元素核对（最快定位）

---

## 7. 和"引擎/训练框架"的关系（面试一句话）

> 训练必须有反向，推理（decode）不需要反向。所以 FlashAttention 反向是「训练侧能力」，PagedAttention 是「推理侧能力」——两边都碰过，才讲得清"为什么推理引擎不实现反向"。

---

## 8. 五个自测问题（学完闭卷答）

1. 前向为什么必须存 $m$ 和 $\ell$，只存 $O$ 行不行？
2. $\delta$ 的两种写法 $\operatorname{rowsum}(dP\odot P)$ 与 $\operatorname{dot}(dO,O)$ 为什么等价？哪个更省？
3. 反向 FLOPs 为什么是前向的 $2.5$ 倍？把每一项拆出来数。
4. $S=4096,\ d=128$ 时 $S\times S$ 矩阵 fp16 占多少？这解释了反向的哪个设计？
5. 为什么 $dQ$ 能寄存器累加、$dK/dV$ 不能？若把循环换成 K/V 外层、Q 内层，谁变成本地、谁变成跨块？

---

## 9. 参考资料

- [FlashAttention-2 (arXiv:2307.08691)](https://arxiv.org/abs/2307.08691) —— §2.3 反向、$\delta$ 推导、并行化策略
- [Triton tutorial `06-fused-attention.py`](https://github.com/triton-lang/triton/blob/main/python/tutorials/06-fused-attention.py) —— 含 `_attn_bwd`，`atomic_add` 版本，可直接对照
- [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) —— CUDA 官方实现（量太大，先别读）
- [FlashAttention 原始论文（含反向推导附录）](https://arxiv.org/abs/2205.14135)

> 顺序建议：**自己按 §2 推公式 → 写 PyTorch 版对数 → 再对照 Triton tutorial**。反过来做会"看懂但写不出"。
