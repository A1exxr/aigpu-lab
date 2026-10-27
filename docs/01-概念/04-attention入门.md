# Attention 入门：从公式到显存流量

> 为 `w4_attn_naive.cu` 准备。读完你要能自己算出"naive 版为什么慢、慢在哪个矩阵"。
> 配套：`kernels/README.md:40`、`ncu速查.md`。

## 0. 一句话

> **naive attention 慢，不是因为算得多，是因为它把 `S×S` 的中间矩阵写进显存又读回来。**
> 这个矩阵的大小**与 head 维度 D 无关**，只跟序列长度 S 有关 —— 所以 S 一长，流量爆炸。

---

## 1. 符号（先把这四个钉死）

| 符号 | 含义 | 本项目取值 | 对应的直觉 |
|---|---|---|---|
| **B** | batch，一次处理几句话 | 4 | 句子数 |
| **H** | 注意力头数 | 8 | 每句话并行几套注意力 |
| **S** | 序列长度（每句几个 token） | 1024 | 句长 |
| **D** | 每个头的维度 | 64 | 每个头自己的向量长度 |

约束：`hidden_size = H × D` = 8 × 64 = 512。

Q、K、V 都是 **4 维张量**：`(B, H, S, D)`。

---

## 2. 公式与形状追踪

```
S_mat = Q · Kᵀ / √D          → (B, H, S, S)     ← 关键：最后一维是 S，不是 D
P     = softmax(S_mat, dim=-1) → (B, H, S, S)
O     = P · V                 → (B, H, S, D)
```

矩阵乘法规则：`(B,H,S,D) × (B,H,D,S)` → 内层 D 消掉 → `(B,H,S,S)`。

**记住形状链**：

```
Q,K,V : (B, H, S, D)
S_mat : (B, H, S, S)   ← S×S，跟 D 无关
P     : (B, H, S, S)
O     : (B, H, S, D)
```

---

## 3. naive 实现的三步（每一步都要碰显存）

| 步骤 | 干什么 | 读 | 写 |
|---|---|---|---|
| ① | `S_mat = Q·Kᵀ/√D` | Q, K | **S_mat（S×S，落显存）** |
| ② | `P = softmax(S_mat)` | **S_mat** | **P（S×S，落显存）** |
| ③ | `O = P·V` | **P**, V | O |

三处 `S×S` 的显存往返：**写 S_mat、读 S_mat、写 P、读 P** —— 一共 **4 次**，每次都是 `B·H·S·S` 个元素。

---

## 4. 把字节数算出来（B=4, H=8, S=1024, D=64, fp16）

先算两个基础量：

```
Q/K/V 各: B·H·S·D = 4×8×1024×64 = 2,097,152 个元素 × 2 B = 4 MiB
S_mat   : B·H·S·S = 4×8×1024×1024 = 33,554,432 个元素
          fp16 → 64 MiB ；fp32 → 128 MiB
```

### naive 版总流量（S_mat 用 fp16）

| 动作 | 字节 |
|---|---|
| 读 Q | 4 MiB |
| 读 K | 4 MiB |
| 写 S_mat | 64 MiB |
| 读 S_mat | 64 MiB |
| 写 P | 64 MiB |
| 读 P | 64 MiB |
| 读 V | 4 MiB |
| 写 O | 4 MiB |
| **合计** | **272 MiB ≈ 285.2 MB** |

**注意 272 MiB 里有 256 MiB 全是那个 S×S 矩阵的往返**，Q/K/V/O 加起来才 16 MiB。

如果 S_mat 用 fp32：合计 **524 MiB**，翻倍。

### 时间下限

```
285.2 MB ÷ 272 GB/s ≈ 1.05 ms     ← 纯带宽下限
```

而真正的浮点计算量：

```
FLOPs = 2·B·H·S·S·D（QKᵀ） + 2·B·H·S·S·D（PV） = 4·B·H·S²·D
      = 4 × 4 × 8 × 1024² × 64 ≈ 8.59 GFLOP
算力下限 = 8.59e9 ÷ 65.2e12 ≈ 0.132 ms
```

**带宽下限 1.05 ms 是算力下限 0.132 ms 的 8 倍** → 时间由显存决定 → **memory-bound**。

---

## 5. 算术强度（AI）：一个数说清"为什么慢"

```
AI = FLOPs ÷ 字节数
naive : 8.59e9 ÷ 285.2e6 ≈ 30 FLOP/B
```

你这张卡的 Roofline 拐点是 **239.7 FLOP/B**（fp16）。**30 远在拐点左边 8 倍** → 必然被带宽卡死。

对比 FlashAttention 的理想情况：

```
只读 Q,K,V 一次（12 MiB）+ 写 O（4 MiB）= 16 MiB
AI = 8.59e9 ÷ 16.78e6 ≈ 512 FLOP/B   → 跑到拐点右边，变成 compute-bound
```

**同一个数学运算，AI 从 30 变成 512 —— 这就是 FlashAttention 的全部魔法。**

---

## 6. FlashAttention 的核心思想（为 w5 铺路）

不把 `S×S` 矩阵写进显存，而是**按块（tile）在片上算完就丢**，用 **online softmax** 边算边更新：

```
对每个 K/V 块（大小 Bc）：
    s   = Q_tile · K_tileᵀ              （只在寄存器/共享内存里）
    m_new = max(m_old, max(s))
    l_new = l_old · exp(m_old - m_new) + Σ exp(s - m_new)
    O     = (O · l_old · exp(m_old - m_new) + exp(s - m_new) · V_tile) / l_new
    m_old, l_old = m_new, l_new
```

三个状态 `m`（行最大值）、`l`（指数和）、`O`（输出累加）**一直留在片上**，永远不落显存。

代价：K/V 会被多个 Q 块重复读（tile 复用的经典权衡），但重复次数由 tile 大小控制，远小于 `S` 倍的 S_mat 往返。

---

## 7. 澄清一个混淆（你问的"残差"）

**残差不在 attention 里。** 你记的那是 w3 做的事：

```
Transformer block（pre-norm 结构）：
    h = x + Attention(LN(x))      ← 这里的 "+ x" 是残差
    out = h + FFN(LN(h))          ← 又一个残差
```

- **残差连接**：`h = x + Sublayer(...)`，属于 block 级别，是**逐元素加法**（你 w3 融合的就是它 + RMSNorm）。
- **attention 内部**：只有 Q、K、V、softmax、P·V，**没有任何残差**。

两者唯一的联系：attention 的输入通常叫 `h`，而 `h` 是上一层残差加出来的结果。**但 memory-bound 的原因是 S×S 矩阵往返，跟残差无关。**

---

## 8. 自测（答得出才能开始写 w4）

1. `S_mat` 的形状是什么？为什么最后一维是 S 而不是 D？
2. B=4, H=8, S=2048, D=64 时，`S_mat`（fp16）是多大？naive 总流量大概多少 MiB？
3. naive 版里，`S×S` 矩阵被写几次、读几次？分别在哪一步？
4. 为什么 AI = 30 就注定是 memory-bound？拐点是多少？
5. FlashAttention 不落 S_mat，那它把什么留在片上？三个状态分别是什么？

---

## 9. 与 w4 的对应关系

你要写的 `w4_attn_naive.cu` 就按三步来：

1. kernel 1：`S_mat = Q·Kᵀ / √D`（grid 覆盖 `B·H·S` 行，每行 S 个元素）
2. kernel 2：按行 softmax（复用你 w2_softmax 的写法）
3. kernel 3：`O = P·V`（每行输出 D 个数，沿 S 累加）

**w4 不要求快。** 它的任务是量出上面那张 272 MiB 的表，并证明"慢在 256 MiB 的往返"。

*建立于 2026-10-19。*
