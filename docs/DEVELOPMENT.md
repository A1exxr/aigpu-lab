# aigpu-lab —— RTX 4060 Laptop（8GB）算子优化项目脚手架

> 目标：六周内产出一个**带真实性能数字**的 AI Infra 项目，
> 让你的简历从"数学建模"变成"算子优化"。

---

## 这是什么 / 不是什么

| | |
|---|---|
| ✅ 是 | 一套**测量工具**：计时、算带宽、验正确性、出报告 |
| ✅ 是 | 一个**流程约束**：强迫你每一步都产出可验证的数字 |
| ❌ 不是 | kernel 代码库。**kernel 必须你自己写**，理由见 `kernels/README.md` |

---

## 放到哪里（重要）

**不要在这个 Windows 目录下编译。** WSL2 访问 `/mnt/e/...` 的 I/O 慢几倍到几十倍。

在 WSL 终端里：

```bash
mkdir -p ~/projects
cp -r /mnt/e/DSH_project/aigpu-lab ~/projects/
cd ~/projects/aigpu-lab

# 用 VS Code 打开它（左下角应显示 WSL: Ubuntu）
code .
```

之后所有工作都在 `~/projects/aigpu-lab` 里做。

---

## 环境准备

先按 `../resume/环境搭建清单-WSL2-CUDA-Profiling.md` 把 CUDA Toolkit + ncu 装好。

```bash
cd ~/projects/aigpu-lab
pip install torch --index-url https://download.pytorch.org/whl/cu124   # 按你的 CUDA 版本调整
pip install numpy matplotlib
```

**自检（这一步不过，别往下走）：**

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -m gpulab.specs        # 打印你的卡规格 + Roofline 拐点
```

第二条命令如果报"规格表里没有匹配"，说明你的卡不在表里 ——
去 NVIDIA 官网查到峰值带宽和 FP16/FP8 Tensor Core 吞吐，补进 `gpulab/specs.py`。
**"先查规格再动手"是第一个练习。**

---

## 快速开始（W1 交付物）

```bash
python run_baseline.py                       # 默认 bs=1 seq=2048 hidden=4096 fp16
python run_baseline.py --seq 4096 --iters 300
python run_baseline.py --no-flush-l2 --out results/baseline_noflush.md   # 对照实验
```

跑完会得到：

1. 每个算子的 **中位数延迟 / P99 / 有效带宽 / 带宽利用率 / 实测算力 / Roofline 判定**
2. attention 的「标准实现 vs torch SDPA」**延迟与峰值显存对比**
3. `results/baseline.md` —— 可以直接贴进 README 的 markdown 报告

### W1 验收标准

- [ ] 能跑出全部数字，没有报错
- [ ] 能解释**为什么要 warmup**（第一次调用的 JIT/autotune/频率爬升会被算进去）
- [ ] 能解释**为什么要 flush L2**（不冲 L2 测的是缓存命中，不是 HBM 带宽）
- [ ] 能解释为什么 `attn_naive` 比 `attn_sdpa` 慢且吃显存
- [ ] 把 `--no-flush-l2` 的结果和默认结果对比，能说出差异的原因

---

## 目录结构

```
aigpu-lab/
├── README.md                       ← 你在这里
├── requirements.txt
├── check_env.py                    ← 环境自检：11 项检查 + 每项的修复命令
├── gpu_watch.py                    ← 边跑负载边采时钟/功耗/温度，检测热降频
├── run_baseline.py                 ← W1：PyTorch 原生算子基线
├── run_attention_scaling.py        ← W4/W5：attention 序列长度扫描（含 OOM 处理）
├── gpulab/                         ← 测量工具（我写的，你不用改）
│   ├── specs.py                      显卡规格 + Roofline + 实测频率重算
│   ├── timer.py                      CUDA event 计时 harness（含 L2 冲刷）
│   ├── workloads.py                  FLOPs / 访存字节数公式
│   ├── verify.py                     正确性校验
│   ├── plotting.py                   Roofline 图 + 伸缩曲线（matplotlib 可选）
│   └── report.py                     markdown 报告生成
├── kernels/                        ← 你写的 kernel（空的，等你去填）
│   └── README.md                     写什么、验收标准
└── results/                        ← 跑出来的报告与图

## 工具速查

| 什么时候用 | 命令 |
|---|---|
| 刚装完环境，不知道还差什么 | `python check_env.py` |
| 跑任何 benchmark（同时监测降频） | `python gpu_watch.py -- python run_baseline.py` |
| W1：拿基线数字 | `python run_baseline.py` |
| W4：拿 attention 的动机数据 + 图 | `python run_attention_scaling.py` |
| W5：把你自己的 FlashAttention 挂进来对比 | `python run_attention_scaling.py --extra "my_flash=kernels.w5_flash_attn:flash_attn_forward"` |

> `run_attention_scaling.py` 会把 OOM 当成**正常数据点**记录下来（而不是崩溃）。
> 在 8GB 卡上「标准实现在 S=8192 直接跑不了」正是你项目最有说服力的一条证据。
```

---

## 六周推进表

| 周 | 做什么 | 交付物 | 验收 |
|---|---|---|---|
| W1 | 环境 + 基线 | `results/baseline.md` | 能独立测出数字并解释方法学 |
| W2 | 手写 CUDA 基础算子 | `kernels/w2_*.cu` | 至少一个算子带宽利用率 > 70% |
| W3 | 算子融合 | `kernels/w3_fused_rmsnorm.cu` | 融合 vs 分步有明确加速比 + 数据流图 |
| W4 | 标准 attention + profile | `kernels/w4_attn_naive.cu` | 用 roofline 证明它是 memory-bound |
| W5 | FlashAttention 前向 | `kernels/w5_flash_attn.py` | 长序列显存 O(S²)→O(S)，延迟对比 |
| W6 | 包装 | README + 博客 | 对比表 + roofline 图 + Nsight 截图 + 优化方向 |

详细说明见 `../resume/项目规格书-RTX4060Ti-6周.md`。

---

## 关于正确性的铁律

**先验正确性，再测性能。永远不要单独报告加速比。**

```python
from gpulab import verify
result = verify.check(ref=torch_ref(x), cand=my_kernel(x), name="my_kernel")
verify.assert_correct(result)     # 不正确直接抛异常，别浪费机时
```

一个算错的 kernel 可以无限快。面试官第一句话一定是"那结果对吗？"

---

## 你的卡的两个关键数字

RTX 4060 **Laptop** GPU（Ada, SM 8.9, 8GB）：

| | |
|---|---|
| 峰值带宽 | **272 GB/s**（4090 是 1008 GB/s，3.7 倍） |
| SM 数 / CUDA 核心 | 24 / 3072 |
| FP16 Tensor | ~58.2 TFLOPS |
| FP8 Tensor | ~116.5 TFLOPS（Ada 支持） |
| **FP4** | **不支持**（需要 Blackwell） |
| 拐点算术强度 | **~214 FLOPs/Byte** |
| TGP | **35–115W（厂商可配置，直接决定性能）** |

**参考对比**（统一按 FP16 累加口径，见 `gpulab/specs.py` 里的说明）：

| 显卡 | 带宽 | FP16 Tensor | 拐点强度 |
|---|---|---|---|
| **RTX 4060 Laptop（你的）** | **272 GB/s** | **58.2 TFLOPS** | **214** |
| RTX 4060 Ti（桌面） | 288 GB/s | 88.3 TFLOPS | 307 |
| RTX 4090 | 1008 GB/s | 330.3 TFLOPS | 328 |
| A100 80G | 2039 GB/s | 312 TFLOPS | 153 |
| H100 SXM | 3350 GB/s | 989 TFLOPS | 295 |

**拐点 ~214 意味着：你日常写的绝大多数算子在这张卡上都是 memory-bound。**
（参考：LayerNorm/Softmax/GELU 的算术强度只有个位数到几十；标准 attention 约 31。）

> ⚠️ **笔记本卡的 TGP 由厂商配置（35W–115W），同一型号不同机器性能可差一倍。**
> 所以你的**绝对延迟只在你机器上有意义**。报告里重点写**加速比**和**带宽利用率**，
> 并把 `power.limit` 记进环境信息。
>
> ⚠️ 另外笔记本卡会**热降频**，长跑 benchmark 数字会飘：必须报告 P99 和标准差，
> 且对比实验要在同一次会话里背靠背跑完。测性能时**插上电源**。

### 这对你来说为什么是好事

1. **memory-bound 优化原理清晰、效果直观**：减少 HBM 往返 → 直接看到加速。面试好讲。
2. **显存只有 8GB，所以你能演示 4090 演示不出来的东西。**
   标准 attention 在 S=8192 时峰值显存 **2088 MB**，torch SDPA 只用 **40 MB**
   —— **相差 52 倍**（实测，见 `results/attention_scaling.md`）。
   在 8GB 卡上这是"还剩多少余量"的问题，比在 24GB 卡上尖锐得多。
3. **笔记本的功耗/散热约束 = 端侧 AI 的真实场景。** 端侧部署工程师每天就是在
   有限功耗、散热、显存预算下做优化 —— 这不是劣势，是叙事资产。
4. **你不需要大算力、不需要多卡，就能做出有竞争力的项目。**

---

## 记住三件事

1. **每个数字都要能解释怎么来的。** 面试官不问你写了多少 kernel，他问你
   `block_size` 为什么是 256。
2. **不要只报延迟，要报加速比和利用率。** 绝对延迟受卡的限制，加速比反映的是你的能力。
3. **WSL2 不支持锁频**，所以每组对比必须在同一次会话里连续跑完，重复 200 次取中位数。
