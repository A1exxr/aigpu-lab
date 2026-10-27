# w7 PagedAttention decode —— 实验记录

> 手动维护。`results/kernels.md` 每次跑 bench 会被整份覆盖，不留历史。

## 测试配置

| 项 | 值 |
|---|---|
| case | B=8, H=32, D=128, L=2048, BS=16, MAXB=128, NB=1024 |
| dtype | fp16 |
| 真实算法流量 | q 64 KiB + KV 读 256 MiB + out 64 KiB ≈ **256 MiB** |
| 实测带宽天花板 | 221.4 GB/s |
| 理论下限 | 256 MiB ÷ 221.4 GB/s ≈ **1.21 ms** |

## 版本台阶

| 版本 | ms | GB/s | 占峰值 | vs 参考 |
|---|---|---|---|---|
| 参考版（`w7_paged_attn_参考版.cu.bak`） | 1.9763 | 135.9 | 50.0% | 4.2× |
| 单遍 online softmax | 1.8897 | 141.9 | 52.2% | 4.4× |
| **2 路展开（ILP-4）** | **1.5084** | **178.1** | **65.5%** | **5.55×** |
| 下限 | 1.21 | 221.4 | 100% | — |

## 正确性

- 收尾块（L 为奇数时处理最后一个 token）已用 **L=2049 PASS** 验证；L=2048 时该分支不执行。
- `w7_paged_attn.cu` 收尾块 4 处修正：`t / BS`、`t % BS`、`d += 32`、`acc[k] = ...`。

## 下一步

1. **split-L**：每个 `(b,h)` 的 1024 次迭代切 4 段给 4 个 block，再用合并 kernel 按 online softmax 规则汇总；grid 32 → 128。
2. `half2` 向量化：dot 循环的 12 次 half load 合成 6 次 32 位 load。
3. 写 README 的 w7 一节（台阶表 + 与 vLLM flash-decoding 的对应关系）。

## 文件

| 文件 | 是什么 |
|---|---|
| `w7_paged_attn.cu` | 当前工作版（2 路展开） |
| `w7_paged_attn_单遍online.cu.bak` | 单遍 online 版（1.8897） |
| `w7_paged_attn_参考版.cu.bak` | 参考实现基线（1.9763） |
