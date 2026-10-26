# INT8 台阶1 实现契约（w8_int8_attn.cu）

REGISTRY 已注册好（`bench_kernels.py` 里 `name="paged_attn_int8"`），
你只需要建 `kernels/w8_int8_attn.cu`，实现下面这个函数，bench 会自动编译它。

## 必须导出的函数

```cpp
torch::Tensor paged_attn_int8(torch::Tensor q, torch::Tensor k_q, torch::Tensor k_scale,
                              torch::Tensor v_cache, torch::Tensor block_table,
                              torch::Tensor context_lens);
```

## 张量契约

| 张量 | 形状 | dtype | 说明 |
|---|---|---|---|
| `q` | `[B, H, D]` | fp16 | 当前 token 的 query |
| `k_q` | `[nb, bs, H, D]` | **int8** | 量化后的 K cache |
| `k_scale` | `[nb, bs, H]` | fp16 | 每个 (token, head) 一个 scale |
| `v_cache` | `[nb, bs, H, D]` | fp16 | V 未量化 |
| `block_table` | `[B, MAXB]` | int32 | 逻辑块 → 物理块 |
| `context_lens` | `[B]` | int32 | 每个 batch 的上下文长度 |
| 返回 | `[B, H, D]` | fp16 | 注意力和输出 |

bench case：`B=8, H=32, D=128, ctx=2048, bs=16, nb=1024, MAXB=128`。

## 核心公式（这条决定性能）

```
score(token t) = scale(t,h) * Σ_d q[d] * k8[t,d]
```

**scale 对 d 是常数，可以提到求和外面**：

```
score = scale * dot(q, k8)      // 归约完之后只乘 1 次
```

所以每个元素只需要：
1. `int8 → float` 一次转换（`(float)k8[d]`，等价 `__int2float_rn`）
2. `q[d] * k8f` 一次乘加

**不要**每元素乘一次 scale（多 128 次乘法），也**不要**先把整行反量化到共享内存（多一次搬运）。

## 改动点（相对 `w7_paged_attn.cu` 单链 2 路展开版）

| 位置 | 原来 | 现在 |
|---|---|---|
| K 指针类型 | `const half*` | `const int8_t*` |
| 每 token 取 scale | 无 | `float s = __half2float(k_scale[(size_t)(phys*BS+off)*H + h]);` |
| 点积内层 | `__half2float(krow[d])` | `(float)krow[d]` |
| 归约后 | `s0` 直接用 | `s0 *= scale0; s1 *= scale1;` |

`k_scale` 的地址算法和 `k_q` 的行首完全同构：把 `D` 换成 `H`。
即 `kq_row = ((size_t)(phys*BS+off)*H + h) * D`，`scale_idx = ((size_t)(phys*BS+off)*H + h)`。

## 验收标准

1. `[PASS]`（`max_abs` 预期从 fp16 的 2.8e-4 涨到 1e-2 量级，贴近容差是**正常的**）。
2. 流量从 256 MiB 降到 **193 MiB**（K 128→64，scale 多 1 MiB），
   带宽下限 **0.92 ms**（对 221.4 GB/s 实测天花板）。
3. 如果耗时 > 1.5 ms，说明反量化/访存把带宽收益吃掉了 → 用
   `--only paged_attn_int8` 看 bench 数字，再决定要不要上 `dp4a`。

## 跑法

```bash
cd ~/projects/aigpu-lab
~/miniconda3/envs/triton/bin/python kernels/bench_kernels.py --only paged_attn_int8
```
