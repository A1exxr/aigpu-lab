# vLLM 在哪一层：从「框架」到「我写的那个 kernel」

> 写这份文档的目的只有一个：**面试官问「你写的 PagedAttention 和 vLLM 差在哪」时，你能答清楚层级。**
> 不是让你去读 vLLM 源码，是让你知道它在系统的哪个位置、和你的 kernel 通过什么接口相接。
>
> 最后校对：2026-10-28。vLLM 迭代很快，**下面的目录名以你拉取时的 `main` 为准**；
> 本文只用我已经核实过的路径（`vllm/v1/core/sched/`、`vllm/v1/core/block_pool.py`），
> 没核实的部分只描述职责，不指路径。

---

## 0. 一句话

**vLLM 是一个 LLM 推理引擎（serving engine）**：它接收 HTTP 请求，把多个用户的不同长度请求
拼成 GPU 上的一批计算，管理每个请求的 KV cache 显存，然后把 token 一个个吐回去。

它不是模型、不是训练框架、不是算子库。它站在**应用**和**GPU** 之间：

```
用户请求 ──► 服务层（OpenAI 兼容 API）──► 推理引擎（vLLM）──► GPU kernel（FlashAttention / GEMM / 我写的那种）
```

---

## 1. 它解决什么问题：朴素推理的三个浪费

自己写一个 `for token in range(max_new_tokens): model(input_ids)` 的循环，会浪费三样东西：

| 浪费 | 朴素做法的问题 | vLLM 的对策 |
|---|---|---|
| **批** | 一个请求跑完再跑下一个，GPU 大部分时间在等 I/O 和 launch | **continuous batching**：每个 decode step 重新组批，谁结束谁滚出 |
| **显存** | 按 `max_seq_len` 预分配 KV cache → 一个 2048 的槽只用 300 也占满 | **PagedAttention**：KV 切成固定大小的 block，按需分配，按页表索引 |
| **重复计算** | 多个请求共享同一段 system prompt，每个都重算一遍 prefill | **prefix caching**：相同前缀的 block 直接复用 |

这三条是 vLLM 的立身之本，也是它相对「HuggingFace `generate()`」的核心优势。

---

## 2. 整体架构：六个角色

```
┌──────────────────────────────────────────────────────────────┐
│ ① API 层（OpenAI 兼容）   /v1/completions  /v1/chat/completions │
│      AsyncLLM：async 接口、流式输出、请求 id 管理                │
└───────────────────────────┬──────────────────────────────────┘
                            │ add_request(request)
┌───────────────────────────▼──────────────────────────────────┐
│ ② EngineCore（引擎核心，跑在独立进程里）                        │
│     step() 是心脏：一次 step = 一次调度 + 一次前向 + 一次回收     │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ ③ Scheduler（调度器）                                   │  │
│  │    · running / waiting 两个队列                          │  │
│  │    · 每个 step 决定：哪些请求进 batch、各给多少 token      │  │
│  │    · 显存不够时 preempt（换出/重算）                       │  │
│  └────────────────────┬───────────────────────────────────┘  │
│  ┌────────────────────▼───────────────────────────────────┐  │
│  │ ④ KVCacheManager（显存管理者）                            │  │
│  │    · BlockPool：所有物理 block 的 free list + 引用计数     │  │
│  │    · 给每个请求分配 block，产出/维护 block_table            │  │
│  │    · 前缀缓存命中判断、block 回收                          │  │
│  └────────────────────┬───────────────────────────────────┘  │
└───────────────────────┼──────────────────────────────────────┘
                        │ SchedulerOutput（本次要算谁、block_table 长什么样）
┌───────────────────────▼──────────────────────────────────────┐
│ ⑤ Worker / ModelRunner（真正调 kernel 的地方）                  │
│    · 把 block_table 拷到 GPU、组 input_ids/slot_mapping        │
│    · 调 attention backend（FlashAttention / FlashInfer / Triton）│
│    · 调 GEMM（prefill 用普通 matmul，decode 用 GEMV 风格）       │
└───────────────────────┬──────────────────────────────────────┘
                        │ logits
┌───────────────────────▼──────────────────────────────────────┐
│ ⑥ OutputProcessor（采样 + 回写）                               │
│    · temperature/top-p/top-k 采样 → token id                   │
│    · 更新请求状态（长度、是否 EOS、是否结束）                    │
│    · detokenize → 流式返回                                     │
└──────────────────────────────────────────────────────────────┘
```

### 各角色职责（面试能说出来就够了）

| 角色 | 一句话职责 | 关键数据结构 |
|---|---|---|
| ① API / AsyncLLM | 收请求、流式回包 | request id → 状态 |
| ② EngineCore | 一个 step 循环，串起调度-执行-回收 | `step()` |
| ③ Scheduler | 决定"这一步算谁" | `running` / `waiting` 队列、token budget |
| ④ KVCacheManager | 决定"KV 放哪、够不够" | `BlockPool`（free list + ref_cnt）、`block_table` |
| ⑤ Worker | 真正在 GPU 上跑 | `block_table` 张量、`slot_mapping` |
| ⑥ OutputProcessor | 采样 + 状态机 | 采样参数、停止条件 |

**你写的 kernel 在 ⑤ 里面，被 ④ 的数据驱动。**

---

## 3. 一次请求的完整生命周期（编号对得上面试）

以「用户发一条 chat 请求，要求输出 100 token」为例：

```
1. API 层收到 POST /v1/chat/completions
2. tokenize → input_ids（比如 512 个 token）
3. add_request：请求进 Scheduler 的 waiting 队列
4. Scheduler.schedule()：
     - 这轮有预算（比如最多 8192 token）
     - 512 个 token 是 prefill，一次能算完 → 取出来
     - 找 KVCacheManager 要 block：ceil(512/16) = 32 个 block
     - 分配到 32 个物理 block id → 拼成 block_table = [7, 9, 12, ...]
5. 产出 SchedulerOutput：{req_id, 要算的 512 token, block_table, is_prefill=True}
6. Worker 准备输入：
     - input_ids 拼成 [512]
     - slot_mapping：每个 token 该写进哪个 KV 槽（由 block_table + 页内偏移算出）
     - block_table 作为 int 张量拷到 GPU
7. 前向：prefill → 得到 512 个位置的 logits，只取最后一个用于采样
8. 采样第 1 个新 token
9. 写 KV：页表第 31 页还有空位 → 不需要新 block
10. 下一轮 step：这个请求进 running 队列，每步只算 1 个 token（decode）
      - 显存检查：KV 长度 513 → 需要 33 个 block；还够 → 继续
      - 每生成 16 个 token 才需要申请一个新 block
11. 某一步显存不够：Scheduler 选择 preempt 某个请求
      - RECOMPUTE：丢掉它的 KV，之后重算 prefill（省显存，费算力）
      - SWAP：把 block 换到 CPU 内存（费带宽）
12. 遇到 EOS 或到 max_tokens：
      - 释放它的全部 block（ref_cnt 归零 → 回到 free list）
      - detokenize → 流式返回 → 请求结束
```

**这张表里有两个东西是你可以直接接上的：**
- 第 4 步产出的 `block_table` —— 你 kernel 的输入；
- 第 6 步的 `slot_mapping` —— 决定新 KV 写到哪。

---

## 4. KV cache 管理：你最该讲清楚的一段

### 4.1 物理布局

```
显存里预先切好 N 个 block，每个 block 装 BLOCK_SIZE 个 token 的 K 和 V：

K_cache: [num_blocks, block_size, num_kv_heads, head_dim]   fp16
V_cache: [num_blocks, block_size, num_kv_heads, head_dim]   fp16

一个 block 的 KV 大小 = 2 * block_size * num_kv_heads * head_dim * 2 字节
例：block_size=16, heads=32, head_dim=128 → 2*16*32*128*2 = 256 KB
```

`num_blocks` 不是随便定的：启动时用「显存总量 − 模型权重 − 激活峰值」除以 block 大小，
**向下取整**，得到的数字就是 `num_gpu_blocks`。这个数字决定了能同时服务多少请求。

### 4.2 block_table：逻辑页 → 物理页

每个请求有一张自己的页表：

```
请求 A 的逻辑序列:  token 0..511  →  逻辑页 0..31
block_table[A] = [7, 9, 12, 15, 3, ...]   # 长度 32，每项一个物理 block id

第 t 个 token 的 KV 位置 = block_table[A][t / 16] * 16 + (t % 16)
```

**注意 `t / 16` 是整除**：页内偏移 = `t % 16`。这就是"分页"的全部含义——
和操作系统的虚拟内存一模一样，只不过页大小是 16 个 token。

### 4.3 引用计数与前缀共享

多个请求可以共享同一个物理 block（相同前缀）：

```
请求 A: [7, 9, 12, ...]   ┐
请求 B: [7, 9, 12, ...]   ┘  前 3 个 block 相同 → ref_cnt = 2
```

- `ref_cnt > 0` 的 block 不能回收；
- 某请求结束时 `ref_cnt -= 1`，归零才回 free list；
- **copy-on-write**：如果 B 要往共享 block 里写新 KV（比如它前缀不同但页内还有洞），
  必须先复制一份 —— 这就是 prefix caching 的代价来源。

### 4.4 preempt 的两种方式

| 方式 | 做什么 | 代价 | 适合 |
|---|---|---|---|
| RECOMPUTE | 直接丢 KV，之后重新 prefill | 费算力（重算前缀） | 默认，decode 阶段 |
| SWAP | block 拷到 CPU pinned memory | 费 PCIe 带宽 | prefill 阶段、重算太贵时 |

这一段能讲出来，就说明你理解"显存不够时引擎怎么办"，而不是只知道"PagedAttention 省显存"。

---

## 5. 调度器：三个必须知道的机制

### 5.1 continuous batching（连续批处理）

不是"一批人齐了才出发"，而是**每个 step 重新组批**：

```
step 1: [A(prefill 512), B(prefill 128)]
step 2: [A(decode), B(decode), C(prefill 300)]     ← C 中途插进来
step 3: [B(decode), C(decode)]                      ← A 结束了，立刻腾出位置
```

**收益**：GPU 利用率从「等人」的 30% 拉到 80%+。
**代价**：每步 batch 形状在变 → 无法用固定 shape 的 CUDA Graph（所以 vLLM 只对 decode 且 batch 稳定时用 graph）。

### 5.2 token budget（这一步最多算多少 token）

调度器不是"能塞多少塞多少"，而是受 `max_num_batched_tokens` 约束：

```
budget = 8192
A 要 512（prefill）→ 剩 7680
B 要 128（prefill）→ 剩 7552
C 要 1（decode）× 剩余 running 请求 30 个 → 30
...
预算用完就停，剩下的排队
```

**为什么要 budget**：一次前向的显存峰值 ∝ 批内 token 总数（尤其 prefill 的激活），
不限制会 OOM。

### 5.3 chunked prefill（分块 prefill）

一个 8K 的 prefill 一次算完会占满预算并阻塞所有 decode（造成 TPOT 抖动）。
chunked prefill 把它切成 512 一段，和 decode 混在一个 batch 里：

```
step k:   [A prefill chunk 0..511, B decode, C decode]
step k+1: [A prefill chunk 512..1023, B decode, C decode]
```

**收益**：TTFT 略增，但 TPOT（每 token 延迟）稳定 —— 生产环境的 SLA 更看重 TPOT。
**这就是"吞吐 vs 延迟"权衡的具体落点**，面试很爱问。

### 5.4 三个延迟指标（一定要分清）

| 指标 | 含义 | 受什么影响 |
|---|---|---|
| **TTFT** | Time To First Token：请求到第一个 token | prefill 计算量、排队长度、chunked 与否 |
| **TPOT** | Time Per Output Token：之后每个 token 的间隔 | decode 的 batch 大小、KV 读取带宽 |
| **吞吐** | tokens/s（全系统） | batch 大小、显存能装多少 KV |

**你做的 kernel 优化（单 token decode 从 1.25 ms → 1.04 ms）直接改善 TPOT。**
这句话就是"我的工作对系统意味着什么"。

---

## 6. vLLM 里和我项目对应的那些优化

| vLLM 的优化 | 我项目里对应 | 我做到什么程度 |
|---|---|---|
| PagedAttention kernel | `w7_paged_attn_split.cu` 等 8 版 | 单请求 decode，97.6% 实测带宽 |
| KV cache 量化（FP8/INT8） | `w8_paged_attn_int8.cu` | 只量化 K：1.0419 ms，87.8% |
| continuous batching | 没做（单请求） | —— |
| prefix caching | 没做（block_table 由我假设给定） | —— |
| CUDA Graph 捕获 decode | 只测过 launch 开销（8.52 µs） | 量过收益边界 |
| 算子融合 | RMSNorm+残差 5N→3N | 92.8% 带宽，1.95× |
| Tensor Core GEMM | wmma 16×16×16 | 0.284 ms，8.5× |

**这张表就是你的定位**：你在「单 kernel 效率」这一格做到了 90%+，
在「调度/批处理/缓存」这些格子里是空白——而那正是 vLLM 的工作。

---

## 7. 我在这套系统的哪个位置

### 7.1 精确到接口

```
vLLM Worker 组 batch
    │  产出: input_ids [num_tokens]
    │        block_table [num_reqs, max_blocks_per_req]  (int32)
    │        seq_lens [num_reqs]
    │        slot_mapping [num_tokens]        ← 新 KV 写到哪
    ▼
attention backend（FlashAttention / FlashInfer / Triton）
    │  对每个 (req, head)：沿 block_table 逐页读 K/V
    ▼
我写的那种 kernel（消费 block_table，逐页 gather + online softmax）
    │  产出: output [num_tokens, num_heads, head_dim]
    ▼
写回 KV cache（按 slot_mapping）
```

**"承接上下"就三句话：**

1. **上游给我什么**：`block_table`（这一页 → 哪个物理 block）、`seq_len`（这个请求多长）、
   `q`（当前 token 的 query）。我不负责分配 block，只负责**按页表读**。
2. **我做什么**：沿 `block_table[0..ceil(L/BS)-1]` 把 K/V 一页一页读进来，
   和 q 做点积、online softmax、累加 V。
3. **下游拿什么**：`[num_heads, head_dim]` 的 fp16 输出，交给下一层（o_proj）。

### 7.2 我的 kernel 假设了什么（面试可主动说的诚实点）

| 我的假设 | 现实里谁负责 | 如果假设不成立会怎样 |
|---|---|---|
| `block_table` 已经给定 | KVCacheManager | 我得自己维护 free list（那是引擎的活） |
| 每个请求的 KV 只追加不修改 | Scheduler（除 copy-on-write） | 共享 block 被改写 → 结果错 |
| batch = 1（单请求 decode） | continuous batching 会给多请求 | 我 kernel 的 grid 要改成 [num_reqs × num_heads] |
| 序列长度固定 2048 | 真实是变长 | 我代码里 `ceil(L/BS)` 已经在处理变长 |

**这段是加分项**：能说清"我做了哪部分、假设了什么、假设由谁保证"，
比"我都做了"可信得多。

---

## 8. 我的项目 vs vLLM：三个不要混淆的口径

| 维度 | 我的 aigpu-lab | vLLM |
|---|---|---|
| 测量对象 | **单个 kernel** 的 ms / GB/s / TFLOPS | **端到端** TTFT / TPOT / 吞吐 |
| 参考基线 | PyTorch 单算子（`F.softmax`、`torch.matmul`） | HF `generate()`、TensorRT-LLM、SGLang |
| 优化目标 | 贴近该 kernel 的 roofline | 相同 SLA 下塞更多请求 |
| 数字口径 | 272 GB/s 标称、221.4 实测 | tokens/s per GPU（受模型、批量、长度影响） |

**所以不要说"我的 PagedAttention 比 vLLM 快 6 倍"。** 正确说法：

> "我在 4060 上把单请求 decode 的 PagedAttention 做到 97.6% 实测带宽，
> 1.04 ms/token。这是**单 kernel** 层面的数字，vLLM 的指标是端到端吞吐，
> 两者不能直接比；但我的 kernel 恰好就是它 attention backend 里那一层。"

---

## 9. 面试四句话（背下来）

1. **vLLM 是什么**：推理引擎，做三件事——continuous batching（每 step 重组批）、
   PagedAttention（KV 按 block 分页）、prefix caching（共享前缀复用）。
2. **它怎么组织**：API → EngineCore（step 循环）→ Scheduler（决定算谁）→
   KVCacheManager（BlockPool + block_table）→ Worker（调 attention/GEMM kernel）→ 采样回写。
3. **我在哪**：我写的是 Worker 里 attention backend 消费的那一层 kernel——
   按 `block_table` 逐页读 K/V 做 decode；我假设页表已给定，它的生产者是 KVCacheManager。
4. **我的数字和它的关系**：我优化的是单 token decode 的访存效率（1.25 → 1.04 ms，
   97.6% 实测带宽），它对应端到端的 TPOT；vLLM 的吞吐指标我这边没有测，也不该直接比。

---

## 10. 五个追问 + 答法

**Q1：为什么 KV cache 要分页，不分页不行吗？**
> 不分页要按 `max_seq_len` 预分配。8 个并发 × 2048 × 32 heads × 128 dim × 2（K/V）× 2 B ≈ 256 MB，
> 而实际平均长度可能只有 300 → 利用率 15%，浪费 85%。分页后按需分配，浪费只剩最后一页的零头（平均半页）。

**Q2：block 大小怎么选？**
> 小 → 内部碎片少，但页表长、gather 次数多、元数据开销大；
> 大 → gather 少、访存连续性好，但每请求浪费半页。
> vLLM 默认 16，是"碎片率"和"访存粒度"的折中。我实测的配置也是 16。

**Q3：prefill 和 decode 的瓶颈一样吗？**
> 不一样。prefill 是大矩阵乘（compute-bound，AI ≈ S/2，S 大时算力瓶颈）；
> decode 是 GEMV 风格（memory-bound，AI ≈ 1，全靠读 KV 的带宽）。
> 所以我的 8 版优化全在 decode 上——它才是延迟敏感的那一半。

**Q4：为什么要 continuous batching，静态 batching 不行吗？**
> 静态批里最短的请求结束后，它占的槽位空转到最长的结束，GPU 利用率崩。
> 连续批处理每步重组，谁结束谁走，利用率从 30% 拉到 80%+。
> 代价是 batch 形状每步在变，CUDA Graph 不好用。

**Q5：你优化到 97.6% 带宽，还有提升空间吗？**
> 单 kernel 没有了（只剩 2.4%）。要再快只能减少字节数：
> 量化（我试过，只量化 K 有效，KV 都量化反而慢——并行度不足）、
> 或者跨 kernel 融合（把 o_proj 融进来）。这两条都是"换赛道"，不是"调参数"。

---

## 11. 术语表

| 术语 | 含义 |
|---|---|
| block / page | KV cache 的分配单位，装 `block_size` 个 token 的 K 和 V |
| block_table | 每个请求的页表：逻辑页 → 物理 block id |
| slot_mapping | 每个新 token 的 KV 该写到哪个具体槽位 |
| ref_cnt | 一个物理 block 被几个请求共享；归零才回收 |
| preempt | 显存不够时把某请求踢出（重算或换出） |
| TTFT / TPOT | 首 token 延迟 / 每 token 延迟 |
| chunked prefill | 把长 prefill 切块，和 decode 混批，稳住 TPOT |
| continuous batching | 每个 decode step 重新组批 |

---

## 12. 怎么自己核对（别信我，信代码）

```bash
# 1. 看调度器真实长什么样（v1 已挪到 sched/ 子目录）
curl -s https://api.github.com/repos/vllm-project/vllm/contents/vllm/v1/core
curl -s https://api.github.com/repos/vllm-project/vllm/contents/vllm/v1/core/sched

# 2. 找 block_table 的定义与形状
curl -s "https://api.github.com/search/code?q=block_table+repo:vllm-project/vllm" | head -50

# 3. 找 attention backend 怎么读 block_table（这一层就是我的 kernel 的位置）
curl -s https://api.github.com/repos/vllm-project/vllm/contents/vllm/v1/attention
```

**只读三个地方就够**（各 20 分钟）：
1. `vllm/v1/core/sched/scheduler.py` 的 `schedule()` —— 看它怎么决定算谁；
2. `vllm/v1/core/block_pool.py` —— 看 free list 和 ref_cnt；
3. `vllm/v1/attention/backends/*` 里任意一个 —— 看 `block_table` 怎么被 kernel 消费（**和你的代码最像**）。

**读的顺序反了会浪费时间**：先看第 3 个（你熟），再看第 2 个（数据怎么来的），
最后看第 1 个（谁决定算谁）。

---

## 13. 一句话收尾

> vLLM 把「多个用户的变长请求」压成「GPU 上连续的批」，我写的是这个批里
> **读 KV cache 的那一步**。它管"算谁、放哪"，我管"读得有多快"。
> 这两件事接口就是 `block_table`。
