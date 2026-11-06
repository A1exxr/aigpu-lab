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
| M2 | 手写 KV cache（不再重算历史） | 同上逐位一致 + 每步耗时随 S **不再线性增长** | 待做 |
| M3 | 接 `paged_attn_split`（分页 KV + 自己的 kernel） | 同上逐位一致 + 吞吐对比 M2 | 待做 |

## 怎么跑

```bash
cd ~/projects/aigpu-lab

# 首次：会下 gpt2（约 500 MB）
export HF_ENDPOINT=https://hf-mirror.com
~/miniconda3/envs/triton/bin/python engines/m1_naive.py

# 已下过：跳过联网检查，秒起
HF_HUB_OFFLINE=1 ~/miniconda3/envs/triton/bin/python engines/m1_naive.py
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
