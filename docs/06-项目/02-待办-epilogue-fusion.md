# 待办：epilogue fusion（GEMM + bias + GELU）

> 状态：**未实现**，排在 mini 引擎 M3 之后。写这份只为不丢想法，不是承诺。
> 预估：2 小时（复用现成代码，不从零写）。

## 要做什么

把你自己的 GEMM 台阶6 kernel 的写回改成“算完直接激活”，不把中间结果落显存。

```
现在（两步）：  h = X·W + bias        → 写 h[M,N] 到显存
                out = gelu(h)         → 读 h，写 out

融合（一步）：  out = gelu(X·W + bias) → 只写 out
```

省掉的是**中间张量的 1 写 + 1 读**。

## 为什么值得做（可量化的预期）

`M = N = K = 1024`、fp16 时：

| 项 | 字节 |
|---|---|
| 读 X、读 W、写 out | 2 + 2 + 2 = **6 MiB** |
| 中间 h 的写 + 读 | 2 + 2 = **4 MiB** |
| 合计（不融合 / 融合） | 10 MiB / 6 MiB |

→ 理论 **1.67×**；实际上 GEMM 是计算型，激活只占一小段，**实测预期 1.15~1.3×**。

## 怎么实现（两步）

1. 复制 `kernels/w6_2gemm.cu` 的台阶6 kernel（`gemm_tc5c` 那条链）到新文件
   `kernels/w6_3gemm_epilogue.cu`；
2. 在写回 `C` 之前，对 `acc` 加 bias 再套 gelu —— **直接搬 `kernels/w2_gelu.cu` 里的 `__half2` 版本**
   （`__half22float2` → 算 → `__float22half2_rn`，全程零 `&`）。

## 验收

| 项 | 标准 |
|---|---|
| 正确性 | 与 `gelu(X@W + bias)`（PyTorch fp32 参考）`max_abs < 1e-2` |
| 性能 | min-of-5 轮 × 100 次，与本文件内交替计时的 cuBLAS 比 |
| 报告 | 与不融合版（GEMM 写 h，再单独 gelu）对比，给出倍数与两个口径（峰值 / cuBLAS） |

## 面试怎么讲

> “epilogue fusion 我做过 A/B：GEMM+bias+GELU 合成一个 kernel，中间张量不落显存。
> 流量从 10 MiB 降到 6 MiB，实测 X 倍。但 GEMM 是计算型，省下的这部分流量只对应一小段，
> 所以收益远低于 RMSNorm+残差那种纯访存融合（那个是 1.95×）。”

**注意别说过头**：这不是“融合一定快”，而是“融合的收益取决于被省掉的那部分占整体多少”。
