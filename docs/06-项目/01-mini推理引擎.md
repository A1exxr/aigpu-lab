# mini 推理引擎：把 6 个手写 kernel 接成一个能跑 batch 的 decode 循环

> 状态：**设计稿（2026-11-05）**，代码未开始。
> 目标读者：三个月后的我自己（忘了为什么要这么设计时回来看），以及面试官（他只看 README 的话，这篇是他会点开的第二篇）。

---

## 0. 一句话

**用你已经写好的 PagedAttention kernel，做出一个"请求进来 → 分批 → 分页 KV → 逐 token 输出"的最小推理引擎，并用吞吐和逐 token 一致性证明它是对的。**

它要证明的是**你知道引擎怎么把多个请求拼进同一个 kernel**，不是"你重写了 vLLM"。

---

## 1. 这是一个项目吗？新开仓库还是放在现在的仓库里？

**结论：是一个独立的工作流（workstream），但不开新仓库。放在 `aigpu-lab/engine/` 下。**

### 为什么不新开仓库

| 理由 | 说明 |
|---|---|
| **一条证据链** | 招聘方点开一个链接，能同时看到"手写 kernel"和"用这些 kernel 拼出的引擎"。两个仓库会让人以为你做了两个半成品。 |
| **复用已有的东西** | `bench_kernels.py` 的 build/计时、`results/` 的口径、`docs/` 的写作规范、`.gitignore`，都已经在了 |
| **内核就是卖点** | 引擎里那次 decode 调用的是**你自己写的 kernel**，这是别人抄不走的；拆成两个仓库反而切断了这个因果链 |
| **成本** | 新仓库要从零配 README、LICENSE、目录、CI 习惯。这些时间对秋招毫无产出 |

### 什么时候才该拆出去

只有一个条件：**引擎做到可以不依赖本仓库、`pip install` 后能跑别人的模型**。现在的目标离这一步很远，所以不拆。

### 目录（新增）

```
aigpu-lab/
├── engine/                       ← 新增，纯 Python
│   ├── README.md                 引擎自己的说明（怎么跑、指标多少）
│   ├── config.py                 EngineConfig：模型路径、块大小、显存预算
│   ├── sequence.py               Request / Sequence 状态机
│   ├── block_manager.py          分页：block_table 的分配、回收、free list
│   ├── cache.py                  KV cache 的物理缓冲 + 写入接口（layout 必须和 kernel 一致）
│   ├── scheduler.py              FIFO + 动态加入 + preempt
│   ├── model_runner.py           模型前向 + 采样，唯一碰 torch 的地方
│   ├── engine.py                 主循环：step() 一次 decode
│   └── tests/
│       ├── test_step1_baseline.py    单请求 vs HF generate
│       ├── test_step2_paged.py       换成 paged cache 后逐 token 一致
│       └── test_step3_batch.py       batch=4/8 vs 逐条，max_abs + 吞吐
└── docs/06-项目/01-mini推理引擎.md    ← 本文
```

`engine/` 只放 Python；kernel 仍然只在 `kernels/`。**引擎通过 `bench_kernels.build_module()` 拿到已编译的算子**，不复制一份 .cu。

---

## 2. 系统全景

```
                      ┌──────────────┐
   请求 ─────────────► │  Scheduler   │  谁这一轮进 batch、谁等着、谁被抢占
                      └──────┬───────┘
                             │ 本轮要跑的 seq 列表 + 每条的 (block_table, context_len, slot)
                      ┌──────▼───────┐
                      │ ModelRunner  │  forward → logits → sample → 下一个 token
                      └──────┬───────┘
                             │ 需要把新 token 的 K/V 写进 cache
                      ┌──────▼───────┐
                      │ BlockManager │  分配/回收物理块，维护 block_table
                      └──────┬───────┘
                             │ 读写
                      ┌──────▼───────┐
                      │  KVCache     │  [NB, BS, H, D] 的平铺缓冲（就是 kernel 吃的那个）
                      └──────────────┘
```

### 一次 `engine.step()` 做什么

```
1. scheduler.schedule()  → 挑出本轮 batch（新请求先 prefill，老请求 decode）
2. block_manager.allocate(seq)  → 给还没块的 seq 分配物理块，追加到 block_table
3. model_runner.forward(batch):
      a. 算 q/k/v（模型权重矩阵乘，PyTorch）
      b. 把 k/v 写进 KVCache 的物理块（按 block_table 找位置）
      c. 调 paged_attn_split(q, k_cache, v_cache, block_table, context_lens) → attn_out
      d. 其余层、lm_head → logits
4. sampler.sample(logits[:, -1])  → 新 token
5. 更新 seq：token 追加、context_len += 1、写没写满一块判断
6. 完成的 seq：回收所有块、出结果
```

**关键点**：`step()` 里**只有一次** decode kernel 调用，batch 里所有请求共用它——这就是"批处理"的全部意义。

---

## 3. 六个核心数据结构

### 3.1 Request（用户提交的）

| 字段 | 类型 | 说明 |
|---|---|---|
| `request_id` | str | 外部 ID |
| `prompt_token_ids` | list[int] | 输入 token |
| `max_new_tokens` | int | 最多生成多少 |
| `sampling_params` | dict | temperature / top_p / greedy |
| `arrival_step` | int | 第几步进来的（模拟并发到达） |

### 3.2 Sequence（引擎内部状态机）

| 字段 | 说明 |
|---|---|
| `status` | `WAITING → RUNNING → FINISHED`（`PREEMPTED` 先不做） |
| `token_ids` | 已经有的全部 token（prompt + 已生成） |
| `num_prompt_tokens` / `num_generated` | 分别多长 |
| `block_table` | `list[int]`，长度 = `ceil(len(token_ids) / BS)`，元素是**物理块号** |
| `last_block_num_tokens` | 最后一块用了几个 slot（决定新 token 写哪） |
| `is_prefill` | 还没跑过前向时为 True |

### 3.3 BlockManager（分页的"操作系统"）

| 字段 | 说明 |
|---|---|
| `num_blocks` | 物理块总数，由显存预算算出 |
| `free_blocks` | `collections.deque`，空闲块号（**用 deque 不用 list**：pop 是 O(1)） |
| `ref_count` | 每个块被几个 seq 引用（prefix caching 才需要；v1 简化直接独占） |

三个方法，**只有这三个**：

```python
allocate(seq) -> None      # 需要新块时从 free 里 pop，append 到 seq.block_table
free(seq) -> None          # seq 结束时把所有块还回去
can_allocate(seq) -> bool  # 调度器用它判断"这块请求现在能不能进"
```

### 3.4 KVCache（kernel 直接吃的那块显存）

```python
k_cache: torch.Tensor   # [num_blocks, BS, H, D]  fp16
v_cache: torch.Tensor   # 同上
```

**layout 不能自己发明**：`w7_paged_attn_split.cu` 的 host 是这么取的

```cpp
int B = q.size(0), H = q.size(1), D = q.size(2);
int MAXB = block_table.size(1);
```

kernel 内部按 `block_table[b, j]` 找到物理块号，再按 `[phys, slot, h, d]` 寻址。所以：

- `q` 必须是 `[B, H, D]`（decode 时每条只有 1 个 token，**batch 维在最左**）；
- `block_table` 必须是 `[B, MAXB]` 的 **int32**，`MAXB = ceil(max_ctx / BS)`；
- `k_cache/v_cache` 必须是 `[NB, BS, H, D]`，**连续**（kernel 用裸指针 + 手算 offset，没有 stride 参数）；
- `context_lens` 是 `[B]` int32，表示每条**总共**看多少个历史 token。

### 3.5 常量约束（来自现有 kernel，改引擎前先看这里）

| 约束 | 值 | 出处 |
|---|---|---|
| `head_dim D` | **≤ 64** | `w7_paged_attn_split.cu` 的 `MAX_D 64` 是编译期常量 |
| `NSPLIT`（split-L 的段数） | 4 | 同上，编译期宏 |
| `BS`（块大小） | 16 | bench case 里的 `bs=16` |
| `H`（q 和 kv 的 head 数） | **必须相同** | kernel 用同一个 `H` 走 q 和 cache，**不支持 GQA** |
| dtype | fp16 | kernel 只吃 `half` |

> **这四条直接决定选哪个模型**，见下一节。

### 3.6 选模型：先 gpt2，不要一上来 Qwen

| 模型 | head_dim | GQA | 能不能直接喂给现有 kernel |
|---|---|---|---|
| **gpt2（124M）** | **64** | 无（MHA，12 头） | ✅ 能，一块都不用改 |
| Qwen2.5-0.5B | 64 | **有**（14 q / 2 kv） | ❌ 要先给 kernel 加"kv head 广播" |
| Qwen3-0.6B | 128 | 有（16/8） | ❌ head_dim 超限，且要改 GQA |

**所以台阶 1–3 全用 gpt2**。理由不是它好，而是**它能把"引擎"和"kernel 改造"这两件事分开**——一次只引入一个新变量。

GQA 留到台阶 4（可选）：给 kernel 加 `num_kv_heads` 参数，`kv_head = h / (H / num_kv_heads)`，改 3 行；那时候你已经有一个能对照的引擎，出错好定位。

---

## 4. 三个可验证台阶

> 原则：**每个台阶结束时，都有一个"和谁比、差多少"的数字**。没有数字的台阶不算完成。

### 台阶 1：单请求基线（纯 PyTorch，不写 CUDA）

**做什么**：加载 gpt2，写一个最笨的循环——每步把整个 `token_ids` 喂进去，取最后一个位置的 logits，argmax，追加。

**为什么要有这一步**：它是后面所有比较的**唯一标准答案**。没有它，台阶 3 出了差异你分不清是"分页错了"还是"kernel 错了"。

```python
# 伪代码
out = model(input_ids)                      # [1, L, V]
next_id = out.logits[0, -1].argmax(-1)
```

**判据**：
- 与 `model.generate(..., do_sample=False, max_new_tokens=20)` 的 20 个 token **逐个相等**；
- 打印每步耗时（这一步慢是正常的，gpt2 全序列重算）。

**产物**：`tests/test_step1_baseline.py`，输出基准 token 序列存成 `results/engine_ref_tokens.json`。

---

### 台阶 2：把 KV cache 换成分页布局（还是不用自己的 kernel）

**做什么**：不再让 HF 自己管 cache，改成**自己维护 `k_cache/v_cache` + `block_table`**，但注意力仍用 PyTorch 算：

```python
# 把 paged cache 还原成连续张量，再算 attention（慢，但正确性一目了然）
k = k_cache[block_table_b].reshape(-1, H, D)[:L]      # [L, H, D]
attn = softmax(q @ k.transpose / sqrt(D)) @ v
```

**为什么可以这么慢**：这一步只验"**分页的地址算得对不对**"，不验性能。用 PyTorch 反而更容易看出索引错。

**判据**：
- 与台阶 1 的 20 个 token **逐个相等**；
- 打印 `block_table` 与 `context_len`，人工抽查一条：第 17 个 token 应该落在 `block_table[1]` 的第 1 个 slot（`BS=16` 时）。

**这一步最容易出的三个错**（先写在这，省得到时候查）：
1. `block_table` 长度用 `ceil(L/BS)`，但**写新 token 时用的是 `L % BS`**，两者别混；
2. 块是**乱序**分配的（`free_blocks` 是 deque），不能假设 `phys == logical`；
3. `context_lens` 是"总长"，不是"这一轮新增长度"。

---

### 台阶 3：批处理 + 换成你自己的 kernel（**这一步才是引擎**）

**做什么**：
1. batch=4 和 batch=8，各条长度不同（10 / 37 / 128 / 300 token），**同时**进入一次 decode；
2. 注意力改成调用 `paged_attn_split`（通过 `bench_kernels.build_module()`）；
3. 测吞吐。

**判据（三条都要）**：

| 判据 | 阈值 | 说明 |
|---|---|---|
| 逐条 token 一致 | 与台阶 2 的逐条结果**完全相同** | 严格相等，不是 max_abs |
| logits 一致 | `max_abs < 1e-2` | 允许 fp16 与 kernel 的路径差异 |
| 吞吐 | `tokens/s(batch=8) / tokens/s(batch=1) ≥ 2` | 低于 2 说明批处理没生效，先查调用次数 |

**这一步的产出是那张表**：

| 配置 | 每步耗时 | tokens/s | KV 显存 | 批内平均长度 |
|---|---|---|---|---|
| batch=1，台阶 2（PyTorch attention） | | | | |
| batch=1，自己的 kernel | | | | |
| batch=4，自己的 kernel | | | | |
| batch=8，自己的 kernel | | | | |

> 面试真正想看的是**最后两行的斜率**：batch 从 1 到 8，时间涨了多少倍、吞吐涨了多少倍。这两个数字的差，就是"引擎把 GPU 喂饱了"的证据。

---

## 5. 调度策略（v1 只做最简单的，但要写清楚）

```python
def schedule(self) -> list[Sequence]:
    batch = []
    # 1. 先让所有 WAITING 的请求尝试 prefill（按到达顺序，FIFO）
    while self.waiting and len(batch) < self.max_batch:
        seq = self.waiting[0]
        if not self.block_manager.can_allocate(seq):
            break                      # 显存不够，这一轮就不收新请求
        self.waiting.popleft()
        self.block_manager.allocate(seq)
        seq.status = RUNNING
        batch.append(seq)
    # 2. 再填满剩下的位置给 RUNNING 的 decode
    for seq in list(self.running):
        if len(batch) >= self.max_batch:
            break
        batch.append(seq)
    return batch
```

**v1 明确不做**（写下来是为了防止自己越写越深）：

| 不做 | 为什么 |
|---|---|
| preempt / 抢占 | 要把已算的 KV 换出到 CPU，v1 没有这个复杂度 |
| 优先级 / 公平性 | FIFO 足够说明"批是怎么形成的" |
| chunked prefill | prefill 一次算完，长 prompt 会拖慢 batch，但这是已知取舍 |
| prefix caching | 需要 `ref_count` + 块哈希，v2 再说 |
| 多卡 / TP | 8 GB 单卡，讲不通的东西不写 |

---

## 6. 指标怎么测（不做这一步，数字就是假的）

| 指标 | 定义 | 怎么测 |
|---|---|---|
| **TTFT**（首 token 延迟） | 提交 → 第一个 token | 在 `step()` 里打 `torch.cuda.Event`，只算 prefill 那一步 |
| **TPOT**（每 token 延迟） | decode 阶段每步平均 | `(总时间 − TTFT) / (总生成 token 数 − 1)` |
| **吞吐** | 总生成 token / 总时间 | 固定总 token 数（比如 8 条各 128 token），跑 3 次取中位 |
| **KV 显存** | `k_cache.numel() * 2 * 2 / 2**20` MiB | 直接算，不查 nvidia-smi |
| **batch 利用率** | 本轮 batch 实际条数 / 上限 | 调度器里打点 |

**三条防坑规则**（都是你这个项目踩过的）：
1. **先 warmup 再计时**（GPU 时钟爬升，第一次必慢）；
2. **同一段代码至少跑 3 次取中位**（你测 cuBLAS 时见过 0.0813~0.0909 的 12% 抖动）；
3. **不许跨口径比**：与 HF 比吞吐时，HF 也要用同样的 `max_new_tokens`、同样的 batch、同样 greedy。

---

## 7. 正确性：三级验证

| 级别 | 比什么 | 阈值 | 抓什么错 |
|---|---|---|---|
| L1 | token id 序列 | **完全相等** | 索引/位置/采样错 |
| L2 | 每步 logits | `max_abs < 1e-2` | fp16 与 kernel 路径差异（可接受） |
| L3 | attention 输出 | `max_abs < 2e-3` | 分页寻址、block_table 错（这一级要单独造数据，见下） |

**L3 单独测的理由**：L1 不一致时你不知道错在注意力还是错在别处。所以给 `paged_attn_split` 做一个**独立单测**：

```python
q = torch.randn(4, 12, 64, dtype=torch.float16, device='cuda')     # B=4, H=12(gpt2), D=64
k_cache = torch.randn(64, 16, 12, 64, ...)                         # NB=64, BS=16
block_table = 随机排列的 [4, 20] int32
context_lens = [10, 37, 128, 300]
# 对照：ref_paged_attn（bench_kernels.py 里已有！直接用）
```

> `kernels/bench_kernels.py` 里的 `ref_paged_attn` 就是现成的 PyTorch 参考实现，**直接 import，不要重写**。

---

## 8. 风险清单（按"最容易炸"排序）

| 风险 | 症状 | 预防 |
|---|---|---|
| **layout 不一致** | 输出全是噪声，但 shape 都对 | 第一件事：写一个 `assert cache.shape == (NB, BS, H, D) and cache.is_contiguous()` |
| **GQA 没处理** | `H` 对不上直接崩，或悄悄算错 | v1 只上 MHA 模型（gpt2），把 GQA 留成台阶 4 |
| **position_ids / rope** | token 都"通顺"但内容错乱 | 每条 seq 自己维护 `position_id = context_len - 1`，绝不能全局共用 |
| **写到错的 slot** | 短序列对、长序列错 | 单测覆盖 `L = 16, 17, 32, 33`（跨块边界） |
| **块泄漏** | 跑几十轮后 OOM | 每次 `free` 后 `assert len(free_blocks) + 在用块数 == num_blocks` |
| **同步开销吃掉收益** | batch=8 却只快 1.2× | 在 `step()` 外面计时，别在 kernel 里面打点 |
| **模型前向成为瓶颈** | 引擎优化不动 | gpt2-124M 在 4060 上单步 ~1 ms，先确认它不是你时间的 90% |

---

## 9. 和 vLLM 的关系（必须能一句话说清）

| 层 | vLLM | 你 |
|---|---|---|
| 模型执行 | 自研 attention backend（FlashAttention/FlashInfer/cutlass） | **你自己写的 `paged_attn_split`** |
| KV 管理 | `BlockPool` + `KVCacheManager`，支持 prefix caching | 简化版 BlockManager（独占、无 prefix） |
| 调度 | continuous batching + preempt + chunked prefill + 优先级 | FIFO + 动态加入 |
| 服务化 | OpenAI 兼容 API、多机、TP/PP | 无（就是一个 Python 类） |

**面试标准答案**：

> "vLLM 我读过分层，没跑通它的全部。我自己实现的是它最内核的两块——分页 KV 的地址计算（在 kernel 里）和 block_table 的生产者（在 Python 里）。调度我用的是 FIFO，没有 continuous batching 的抢占，因为单卡 8 G 上先把'批怎么形成'跑对更重要。"

---

## 10. 面试怎么讲（三句 + 五个追问）

**三句**：
1. "我写了一个 mini 推理引擎：请求进来后由 BlockManager 分页、Scheduler 组批，decode 时**整个 batch 只调一次**我自己写的 PagedAttention kernel。"
2. "它是三层验证过的：token 与 HF 逐位一致、logits max_abs < 1e-2、注意力单测对 PyTorch 参考 < 2e-3。"
3. "batch 1→8 时每步耗时只涨 X 倍，吞吐涨 Y 倍；KV 显存按块精确控制，没有内部碎片。"

**五个追问（提前准备）**：

| 追问 | 答法 |
|---|---|
| 为什么用分页而不是预分配 max_len？ | 预分配浪费 = `(max_len − L)/max_len`，一条 2048 上限、实际 100 token 的请求浪费 95% |
| 块大小 16 怎么定的？ | 太大→内部碎片；太小→block_table 变长、kernel 里循环次数多。16 × (H·D·2B) 正好是一行 cache line 组的规模 |
| batch 里长度不齐，kernel 怎么处理？ | `context_lens` 逐条判断，超出部分 `continue`；这正是 split-L 版本相比 naive 的优势 |
| 为什么不做 continuous batching 的抢占？ | 抢占要把 KV 换出/换入，收益在多请求长尾时才有；单卡 8 G 先保证不 OOM、批能形成 |
| 你这个和 vLLM 差多少？ | 吞吐差 N 倍（如果测了就说），功能上少 prefix caching / 抢占 / TP；但**分页寻址和 kernel 是我自己写的** |

---

## 11. 里程碑（按台阶，不按天数）

| # | 交付物 | 完成标志 |
|---|---|---|
| M1 | `tests/test_step1_baseline.py` + `results/engine_ref_tokens.json` | 与 HF 20 token 一致 |
| M2 | `block_manager.py` + `cache.py` + `test_step2_paged.py` | 与 M1 逐 token 一致 |
| M3 | `scheduler.py` + `engine.py` + `test_step3_batch.py` | batch=4/8 一致 + 吞吐表 |
| M4 | `engine/README.md` + 本文状态改成"已完成" | 别人照 README 能跑出同样的数 |

**每完成一个 M，就 commit 一次，并在 commit message 里写当次的数字**（这个项目最值钱的就是"每一步都有数"）。

---

## 12. 参考资料

- vLLM V1 架构总览（调度器/BlockPool 的分层）：[vLLM V1: A Major Upgrade to vLLM's Core Architecture](https://vllm.com.cn/blog/2025-01-27-v1-alpha-release)
- vLLM 系统剖析（continuous batching、PagedAttention 的动机与数据）：[走进 vLLM：高吞吐量 LLM 推理系统的剖析](https://vllm.com.cn/blog/anatomy-of-vllm)
- 一个 1200 行的最小实现，可作为"结构对照"而不是"照抄对象"：[nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm)（同样在 8 GB 笔记本卡上跑 Qwen3-0.6B，报告 1434 tokens/s）
- PagedAttention 的原始动机（KV 浪费 60–80%）与块表设计：见上述剖析文的 PagedAttention 小节

> 读别人的实现只用来**对齐结构**（谁是 block 的生产者、谁是消费者）。**不要抄代码**——你的 kernel 接口和它们不一样，抄了反而要改更多。

---

## 13. 现在的位置

- ✅ 已有：`paged_attn_split` kernel（1.2483 ms、97.6% 实测带宽）、`ref_paged_attn` 参考实现、`build_module()` 编译链、`results/` 口径
- ⬜ 未开始：`engine/` 下所有文件
- 下一步：**M1**。纯 PyTorch，不碰 CUDA，跑通"单请求 20 token 与 HF 一致"再说别的。
