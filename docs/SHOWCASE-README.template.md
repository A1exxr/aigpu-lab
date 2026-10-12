<!--
================================================================================
 README 展示版模板
================================================================================
 这是给**面试官**看的 README，和 README.md（给你自己看的工作手册）不是一回事。

 区别：
   - README.md 回答"我怎么用这套工具" → 结构、命令、参数
   - 这份回答"这个人做了什么、结果如何、他懂不懂" → 结论先行、有数字、有图

 什么时候用它：
   W6 定稿时，把本文件的内容复制进 README.md 覆盖掉，
   工作手册的内容改名为 docs/DEVELOPMENT.md 保留。

 写作纪律（很重要）：
   · 所有【】都要换成真实数字。**没有数字的部分整节删掉，不要留空占位。**
   · 结论先行。面试官平均只花 40 秒看一份 README。
   · 主动写出限制和没做到的地方 —— 这比全篇漂亮话更显专业。
================================================================================
-->

# 在 8GB 笔记本显卡上做 Transformer 算子优化

> 在 RTX 4060 **Laptop**（8GB 显存，272 GB/s 带宽，功耗受限）上从零手写 CUDA / Triton
> 算子，通过 memory-bound 优化与 FlashAttention 融合，
> 把 **【算子名】** 的延迟降低 **【?】x**，把 attention 的显存占用从 O(S²) 降到 O(S)——
> 把序列长度 8192 下的 attention 峰值显存从 **2088 MB 降到 【?】 MB**（**【?】 倍**）——
> 在只有 8GB 显存的卡上，这个差距是"还剩多少余量"的问题。

<!-- 徽章可选，但能让页面看起来像样 -->
![platform](https://img.shields.io/badge/GPU-RTX%204060%20Laptop%208GB-76B900)
![cuda](https://img.shields.io/badge/CUDA-12.8-76B900)
![license](https://img.shields.io/badge/license-MIT-blue)

---

## 结果一览

<!-- 这是全篇最重要的表格。面试官只会看这里。 -->

| 算子 | 实现 | 延迟 (ms) | 有效带宽 (GB/s) | 带宽利用率 | 相对 PyTorch |
|---|---|---|---|---|---|
| RMSNorm | PyTorch | 【?】 | 【?】 | 【?】% | 1.00x |
| RMSNorm | **本仓库 CUDA** | 【?】 | 【?】 | 【?】% | **【?】x** |
| Softmax | PyTorch | 【?】 | 【?】 | 【?】% | 1.00x |
| Softmax | **本仓库 CUDA** | 【?】 | 【?】 | 【?】% | **【?】x** |
| RMSNorm+残差 | 分步（两次 kernel） | 【?】 | 【?】 | 【?】% | 1.00x |
| RMSNorm+残差 | **本仓库融合 kernel** | 【?】 | 【?】 | 【?】% | **【?】x** |

### Attention：显存从 O(S²) 降到 O(S)

| 序列长度 | 标准实现峰值显存 | 本实现峰值显存 | 标准实现延迟 | 本实现延迟 |
|---|---|---|---|---|
| 1024 | 【?】 MB | 【?】 MB | 【?】 ms | 【?】 ms |
| 2048 | 【?】 MB | 【?】 MB | 【?】 ms | 【?】 ms |
| 4096 | 【?】 MB | 【?】 MB | 【?】 ms | 【?】 ms |
| **8192** | **2088 MB** | **【?】 MB** | **31.5 ms** | **【?】 ms** |

![显存对比](results/fig_attention_memory.png)
![Roofline](results/fig_roofline_attention.png)

---

## 硬件平台与它的限制

先把环境说清楚，因为**这张卡决定了整个优化的方向**。

| 项 | 数值 |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU（Ada, SM 8.9） |
| SM 数 / CUDA 核心 | 24 / 3072 |
| 显存 / 带宽 | 8GB / **272 GB/s** |
| FP16 Tensor 峰值 | 【?】 TFLOPS（**用实测持续频率重算**，不是厂商标称值） |
| 实测持续 SM 时钟 | 【?】 MHz |
| 功耗墙 | 【?】 W（Default 【?】 W —— 电源模式变化会降频） |
| 拐点算术强度 | **【?】 FLOPs/Byte** |

**两个直接影响结论的事实：**

1. **这是一张严重"缺带宽"的卡。** 拐点算术强度高达 【?】 FLOPs/Byte，而
   LayerNorm / Softmax / 标准 Attention 的算术强度只有个位数到几十 ——
   **绝大多数算子在这里都是 memory-bound**，优化方向必须是"减少 HBM 往返"，
   而不是"堆算力"。
2. **它是笔记本卡，会热降频。** 我用自写的采样工具记录了跑 benchmark 期间的
   持续频率，前 25% 与后 25% 的中位时钟比值为 **【?】**（< 0.92 即为明显降频）。
   因此所有对比实验都在同一次会话里背靠背跑完，并报告中位数与 P99。

---

## 我做了什么

### 1. 先建立可信的测量基线

优化之前先确保**测量本身是对的**，否则后面所有数字都是错的。这一步解决了三个坑：

- **异步执行**：CUDA 调用是异步的，用 `time.time()` 测到的是"CPU 发命令的时间"。
  改用 CUDA event 计时。
- **L2 缓存污染**：反复测同一份数据会命中 L2，测出来的是缓存速度不是显存带宽。
  每次计时前冲刷 L2（256MB 缓冲区）。
- **峰值算力口径**：`nvidia-smi` 的 `Max Clocks 3105 MHz` 是硬件上限，
  厂商标称 boost 2370 MHz 也不等于实际持续频率。我用采样工具实测持续 SM 时钟后
  **重算**了峰值算力。

### 2. 手写 CUDA 算子（不抄教程）

实现 RMSNorm / Softmax / GELU，并对每个算子回答了这几个问题：

- `block_size` 为什么选 【?】？改成 128 / 512 后性能如何变化？（实测：【?】）
- 用了什么规约策略？（warp shuffle / 共享内存 / 树形规约）
- 共享内存用了 【?】 字节，occupancy 是 【?】%
- 有效带宽达到峰值 272 GB/s 的 **【?】%**
- 是否存在 bank conflict / warp divergence？如何验证的？

### 3. 算子融合：memory-bound 优化的核心

把 `RMSNorm + 残差` 从两个 kernel 融合成一个。

| | HBM 访存 | 延迟 |
|---|---|---|
| 分步实现 | 读 x、读残差、写中间结果、再读、再写 = **【?】N** | 【?】 ms |
| 融合实现 | 读 x、读残差、写输出 = **3N** | 【?】 ms |
| **节省** | **【?】%** | **【?】x** |

```
分步：  x ──┐
            ├─►[add]─► HBM ──►[norm]─► HBM
       res ─┘                     ↑ 多一次往返

融合：  x ──┐
            ├─►[ fused kernel ]─► HBM  ← 只读写一次
       res ─┘
```

### 4. FlashAttention：把瓶颈从带宽转移到算力

用 Triton 实现 FlashAttention 前向。核心思路是**用重算换访存**：

| | FLOPs | HBM 访存 | 算术强度 | 瓶颈 |
|---|---|---|---|---|
| 标准 attention | 【?】 G | **【?】 MB** | 【?】 | memory-bound |
| FlashAttention | 【?】 G（**相同**） | **【?】 MB** | 【?】 | compute-bound |

> **FlashAttention 没有减少任何计算量**，它减少的是 HBM 流量。
> 流量下降 【?】 倍后，算术强度越过拐点，瓶颈从带宽转移到算力。

---

## 关键发现

<!-- 这一节是区分"复现"和"真做过"的地方。写你自己的观察，不要抄。 -->

1. **【?】**（例：在 memory-bound 算子上，float4 向量化访存带来的收益远大于提高 occupancy）
2. **【?】**（例：8GB 显存下真正稀缺的是显存而不是算力，这决定了分块策略）
3. **【?】**（例：理论加速比 6.8x，实际只拿到 X x，差距来自 Tensor Core 利用率只有 Y%）

---

## 复现方式

```bash
# 环境：WSL2 + CUDA 12.8，RTX 4060 Laptop 8GB
python check_env.py                       # 自检环境
python gpu_watch.py -- python run_baseline.py       # 基线 + 降频监测
python kernels/bench_kernels.py                     # 编译并测量自写算子
python run_attention_scaling.py                     # attention 扫描 + 出图
```

所有结果写入 `results/`，其中 `results/*.md` 是完整报告。

---

## 项目结构

```
├── run_baseline.py            PyTorch 原生算子基线
├── run_attention_scaling.py   attention 序列长度扫描（含 OOM 处理）
├── gpu_watch.py               跑负载时的时钟/功耗/温度采样（检测热降频）
├── check_env.py               环境自检
├── gpulab/                    测量工具（计时 / Roofline / 带宽 / 正确性 / 画图）
├── kernels/                   手写 CUDA / Triton 算子
└── results/                   报告与图表
```

---

## 已知限制

<!-- 主动写。面试官会问"你的方案有什么问题"，你要有准备。 -->

- 单卡，**没有多卡 / 分布式经验**。分布式部分我只在单机上用 gloo 后端模拟过
  通信原语，并读过 Megatron-LM 的张量并行实现，没有实机验证过。
- 笔记本 GPU 的持续频率受功耗墙与散热影响，**绝对延迟不具备跨机器可比性**；
  可比的是加速比与带宽利用率。
- FlashAttention 只实现了前向，**反向未实现**。
- 精度实验只做到 FP8（Ada 不支持 FP4）。
- 【?】（你自己的观察）

---

## 参考

- [FlashAttention 论文](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2](https://arxiv.org/abs/2307.08691)
- [Triton 官方教程](https://triton-lang.org/main/getting-started/tutorials/index.html)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
