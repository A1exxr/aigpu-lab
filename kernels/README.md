# 这个目录是留给你的

**`kernels/` 里的每一个 kernel，都必须你自己从零写。** 我不会替你写。

## 为什么

你的瓶颈不是"不会写 CUDA"，是**"只会复现，不会从零"**——就是你自己问的那句
"什么叫从零开始写？"

从零 = 不查教程、不看别人的 kernel，自己推导索引和内存访问，并且**能解释每一个
数字为什么是这个值**（block_size 为什么是 256、tile 为什么是 128×128、共享内存
为什么用这么多）。

如果我替你把 kernel 写了，你这个项目就白做了——面试官两个问题就能问穿。

## 我替你写好了什么

外围的样板工程，这些不体现你的能力，所以我来干：

| 文件 | 作用 |
|---|---|
| `gpulab/timer.py` | CUDA event 计时（warmup / L2 flush / 取中位数） |
| `gpulab/specs.py` | 显卡规格 + Roofline 拐点计算 |
| `gpulab/workloads.py` | 访存字节数 / FLOPs 公式 |
| `gpulab/verify.py` | 和 PyTorch 对比正确性 |
| `gpulab/report.py` | 生成 markdown 报告 |
| `run_baseline.py` | W1 的基线测量脚本（已可直接运行） |

## 你要写什么

按周推进，每个文件都要能独立跑：

```
kernels/
  w2_reduce.cu          # 规约求和：先写最朴素的，再写 warp shuffle 优化版
  w2_softmax.cu         # softmax：memory-bound 的典型
  w2_rmsnorm.cu         # RMSNorm：纯 memory-bound，用来练带宽利用率
  w2_gelu.cu            # element-wise：练 float4 向量化访存
  w3_fused_rmsnorm.cu   # 融合：RMSNorm + 残差（+ 量化），对比分步实现
  w4_attn_naive.cu      # 标准 attention：用来证明"它是 memory-bound"（动机）
  w5_flash_attn.py      # FlashAttention 前向（Triton 版上手更快）
  w6_gemm.cu            # GEMM 三台阶：naive → shared tiling → 4×4 寄存器分块（→ Tensor Core）
  w7_paged_attn.cu      # PagedAttention decode：单链 2 路展开（对照版）
  w7_paged_attn_split.cu # PagedAttention decode：split-L 四段并行（当前最优）
  bench_kernels.py      # 复用 gpulab，测你自己的 kernel
```

## 现状（2026-10-26）

| kernel | 状态 | 关键数字 |
|---|---|---|
| `w2_reduce.cu` | 完成 | 0.1137 ms / 54.3% 带宽，ncu DRAM 87.3%（已贴墙） |
| `w2_softmax.cu` | 完成 | 0.2160 ms / 57.1%，屏障 20 → 4 |
| `w2_gelu.cu` | 完成 | 0.1516 ms / **81.4%**，`__half2` 向量化，追平 PyTorch |
| `w2_rmsnorm.cu` | **跳过** | 知识点已被 w3 融合版覆盖 |
| `w3_fused_rmsnorm_residual.cu` | 完成 | 融合版 199.9 µs / ncu DRAM 92.8%，比两 kernel 版快 1.95× |
| `w4_attn_naive.cu` | 完成 | 46.87 → **15.53 ms**（warp-per-j + PV 四路切分 + QK ILP-4） |
| `w5_flash_attn.py` | 完成 | **0.474 ms**，达 PyTorch SDPA 的 91%，naive 的 1/32.8 |
| `w6_gemm.cu` | **台阶 1–4 完成** | 2.42 → 1.78 → 0.657 → **0.2836 ms**（Tensor Core），超 PyTorch fp32 基线 1.13×；64×64 tile 反例更慢（0.3758） |
| `w7_paged_attn_split.cu` | 完成（已封板） | 2.3644 → 1.8897 → 1.5084 → **1.2483 ms**，215.1 GB/s = **97.2% 实测天花板** |

## 每个 kernel 的验收清单

写完一个 kernel，必须能回答这五题（答不上来就等于没写完）：

1. `block_size` 为什么是 256？改成 128 或 512 后性能怎么变？**测出来给我看。**
2. 用了什么规约策略？warp shuffle / 共享内存 / 树形规约？
3. 共享内存用了多少字节？occupancy 是多少？
4. 有效带宽达到峰值带宽（**272 GB/s**）的百分之多少？
5. 有没有 bank conflict / warp divergence？怎么验证的？

## 用 gpulab 测你自己的 kernel

```python
import torch
from gpulab import specs, timer, verify, workloads

spec = specs.detect()
x = torch.randn(1, 2048, 4096, device="cuda", dtype=torch.float16)

# 1) 先验正确性 —— 不正确就不准谈性能
result = verify.check(
    ref=torch.nn.functional.layer_norm(x, (4096,), w, b),
    cand=my_kernel(x, w, b),
    name="my_rmsnorm",
)
verify.assert_correct(result)

# 2) 再测性能
stats = timer.bench(lambda: my_kernel(x, w, b), warmup=50, iters=200)
nbytes = workloads.bytes_layer_norm((1, 2048, 4096), torch.float16, affine=False)
bw = workloads.bandwidth_gbps(nbytes, stats["median_ms"])
print(f"中位数 {stats['median_ms']:.4f} ms   有效带宽 {bw:.1f} GB/s "
      f"({100*bw/spec.bw_gbps:.1f}% of peak)")
```

## 编译 CUDA 的建议

用 PyTorch 的 `load_inline`，不用自己写 CMake：

```python
from torch.utils.cpp_extension import load_inline

mod = load_inline(
    name="my_kernels",
    cpp_sources="torch::Tensor my_rmsnorm(torch::Tensor, torch::Tensor);",
    cuda_sources=open("kernels/w2_rmsnorm.cu").read(),
    functions=["my_rmsnorm"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
    verbose=True,
)
```

`-lineinfo` 是给 Nsight Compute 用的，能让你在 profiler 里看到源码行号。

---

## 提醒

**每一步都要跑出数字再往下走。** 六周结束时，你的 README 里必须有：

- 一张基线 vs 优化后的对比表（含加速比）
- 一张 Roofline 图（优化前后的点在图上的移动）
- 一张 Nsight Compute 截图
- 一节"下一步还能怎么优化"

有这四样，你就是第 3 级；没有，你就是第 1 级。
