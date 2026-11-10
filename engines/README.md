# engines/ — 手写 mini 推理引擎

> 目的：把 `kernels/` 里那些**单算子**串成一个**能跑、能对账、有数字**的推理引擎。
> 每上一个台阶都有一条判据，过不了就不进下一步。

## 为什么要这一层

`kernels/` 里的 attention kernel 吃的是 `block_table`、`context_lens`——**但没人生产它们**。
bench 里这些是 `torch.randn` 造的假数据；真实系统里，`block_table` 是 BlockManager 分配、
随请求增长的、块用完要回收的。

这个目录补的就是那个"生产者"：**分页 KV 管理 + 调度 + 采样**，然后把 decode 交给自己写的 kernel。

做完之后，项目从"我实现了一个 PagedAttention kernel"变成
"我实现了一个推理引擎的两半：**生产者**和**消费者**"。

## 台阶与判据

| 台阶 | 内容 | 判据 | 状态 |
|---|---|---|---|
| **M1** | 纯 PyTorch 手写 greedy 循环（每步重算整段） | 20 token 与 HF `generate` **逐位一致** | ✅ 已通过 |
| **M2** | 手写 KV cache（不再重算历史） | 同上逐位一致 + 每步耗时随 S **不再线性增长** | ✅ 已通过 |
| **M3a** | 分页 KV 池 + block_table（attention 仍由 HF 算） | 8 步 × 12 层 K/V 与 HF **逐位一致** | ✅ 已通过 |
| **M3b-1** | cache 全自己管：池取 K/V + 自己算 attention（仍是 PyTorch） | 8 步 greedy token 与 M2 **完全一致** | ✅ 已通过 |
| **M3b-2** | 把 SDPA 换成 `paged_attn_split` kernel（fp16） | token ≥19/20 + 单步耗时对比 HF attention | ✅ 已通过（0.93×，S=13） |
| **M4** | 分页 KV 读写改成索引 kernel + 报 TTFT/TPOT/吞吐（`m4_metrics.py`） | token 20/20 + TTFT ≤ 25 ms | ✅ 已通过（TTFT 120.13 → 7.37 ms） |

## 怎么跑

```bash
cd ~/projects/aigpu-lab

# 首次：会下 gpt2（约 500 MB）
export HF_ENDPOINT=https://hf-mirror.com
~/miniconda3/envs/triton/bin/python engines/m1_naive.py

# 已下过：跳过联网检查，秒起
HF_HUB_OFFLINE=1 ~/miniconda3/envs/triton/bin/python engines/m1_naive.py

# M2：5 轮交替测量（丢首次 + 剔 >2× 中位 + 中位数）
HF_HUB_OFFLINE=1 ~/miniconda3/envs/triton/bin/python engines/m2_kvcache_bench.py
```

依赖：`transformers`、`torch`（conda env `triton` 里已装）。

## M1 的实测结果（RTX 4060 Laptop，gpt2 124M，fp32）

```
prompt : 'The capital of France is'
我的   : ' the capital of the French Republic, and the capital of the French Republic is ...'
HF     : （完全一致）
逐位一致: True
手写循环总耗时: 730.4 ms      # 20 步
峰值显存: 496.0 MB
```

关键观察：**每步越来越慢**——第 1 步 S=6，第 20 步 S=26，因为每步都把整段重新 forward，
累计做功 O(S²)。这就是 M2 用 KV cache 要消掉的浪费。

## M2 的实测结果（gpt2 124M，fp32，batch=1）

```
prompt S = 441；每轮 重算 20 步 → 增量 20 步，共 5 轮交替
逐位一致: True（5 轮全部相同）

                  中位数        范围
重算 20 步总耗时   355.03 ms     351.40–367.17
增量 20 步总耗时   110.08 ms     105.11–123.62   → 3.23x
重算 稳态单步       18.57 ms      17.32–19.04
增量 稳态单步        4.89 ms       4.44– 5.86   → 3.79x

峰值显存 704.5 MB
```

两条结论：

1. **稳态单步 3.79×**：KV cache 消掉的是"每步重算整个前缀"，所以越到后面越省；
   S 越大省得越多，这是 M3 上分页的动机。
2. **总耗时只有 3.23×**：因为第 1 步要吃完整 prompt（prefill），两条路径都省不掉；
   20 步太短，摊不薄 prefill。**对外分别报稳态和总耗时两个数，不要混用。**

测量协议（可复现）：交替测 A/B（机器频率与温度漂移对两边公平）→ 丢弃第 1 轮冷启动
（第一次走 `use_cache=True` 要建 cache、扩分配器，实测是 110 ms 的 15 倍）→ 剔除
偏离中位数 2 倍以上的样本 → 取中位数。脚本 `m2_kvcache_bench.py` 自带这些步骤，
并每轮 `assert` 两种实现输出逐位相同。

## M3 的实测结果（gpt2 124M，fp32，batch=1，S=12 → 19）

```
M3a   8 步 × 12 层 K/V 与 HF DynamicCache 逐位相同        ← 分页语义对
M3b-1 8 步 greedy token 与 M2 完全一致: True              ← cache 换掉也对
池显存 72.0 MiB（容量 1024 token），本请求 2 块装 19 token
```

两个**不报错但结果错**的坑（都已修，值得记）：

1. `patch_attention` 里直接 `for layer_idx ...: def patched(...)` → Python 闭包捕获的是
   变量名，循环结束后 12 个 patched 全看到 `layer_idx=11`。必须用工厂函数绑参数。
2. 我们自己管 cache，就必须自己传 `position_ids`：传 `past_key_values=None` 时 HF 认为
   没有历史，每步位置都从 0 开始算 → 第一个 token 对、后面全崩（`[1849, 11, 11, ...]`）。

## 符号表（全项目统一，与 `kernels/` 的 CUDA 代码一致）

| 符号 | 含义 | gpt2 |
|---|---|---|
| `B` | batch | 1 |
| `S` | 序列长度（HF 文档里叫 `T`，我们用 `S`） | 6 → 26 |
| `D` | head 维度 | 64 |
| `H` | head 数 | 12 |
| `L` | 层数 | 12 |
| `V` | 词表大小 | 50257 |

形状：`input_ids [B,S]`、`logits [B,S,V]`、`KV cache [B,H,S,D] × L × 2`。

## 明确不做

- 不做 sampling（top-k / top-p / temperature）——greedy 就够验证正确性
- 不做 beam search、不做 batch>1 的调度（M3 之后再谈）
- 不复刻 vLLM：不碰 prefix caching、preemption、chunked prefill
- 不追求比 HF 快多少——先要"对得上"，再谈"快多少"

## M4 的实测结果（RTX 4060 Laptop，gpt2 124M fp16，prompt 441 token，生成 20 步）

3 轮交替测量（HF / 我们 各 3 轮，丢弃第 1 轮冷启动，取中位数）：

| 指标 | 改前 | 改后 | HF 参考 |
|---|---|---|---|
| TTFT | 120.13 ms | **7.37 ms**（7.18–7.39） | 7.33–7.87 |
| TPOT | 4.73 ms | **4.67 ms**（4.67–5.04） | 4.06–4.38 |
| 吞吐 | 94.4 tok/s | **204.2 tok/s** | 223–234 |
| 总耗时 | 212 ms | **96.1 ms** | 86.4 ms |
| token 一致 | 20/20 | 20/20 | — |

## M4 改了什么（以及为什么这么改）

M3b-2 的 append/gather 是**逐 token 的 Python 循环**：

```python
# 改前：append 每层 441 次调用，每次 2 次 slice-assign → 全模型 10 584 次
for j in range(s_new):
    self.pool.write_token(layer, blk, off, k_new[:, j], v_new[:, j])

# 改后：每层一次 index_copy_ → 全模型 24 次
self.pool.write_run(layer, idx, k_new.transpose(0, 1), v_new.transpose(0, 1))
```

```python
# 改前：gather 逐 token 索引 + stack(441 个 [H,D])
k = torch.stack(ks, dim=1)

# 改后：一次 index_select
k = kf.index_select(0, idx).transpose(0, 1).contiguous()
```

- 关键观察：池里一层 `[64, 16, 12, 64]` 摊平就是 `[1024, 12, 64]`，**每个 token 占一行**，
  于是"把 S 个 token 写进各自的块"就是一次行索引 scatter。
- `idx` 与层无关，**只在 layer 0 算一次**，12 层复用。
- decode（`s_new == 1`）走回原来的直写：24 次 `index_copy_(1 行)` 比 24 次 slice-assign 贵约 0.2 ms/步。

**结论：瓶颈是 Python 调用次数（一次 prefill 约 15 000 次），不是拷贝字节数。**
往返本身不贵——B′（prefill 不 gather，直接用连续 K/V）最终**没有做**，因为已经把 TTFT 压到 HF 以下。

**记账不做**：decode 每步仍比 HF 慢 0.52 ms（19 步共 9.7 ms），来源是
12 层各一次 pybind 调用 + `paged_attn_split` 的寻址开销。要消掉需要 CUDA Graph 捕获
或把 12 层合成一个 kernel，成本高，记在账上不排期。
