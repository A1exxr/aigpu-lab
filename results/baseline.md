# PyTorch 原生算子基线（W1）

> 生成时间：2026-10-11 22:42:01

## 环境

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU (8GB, Ada, SM 8.9, 24 SM)
- 峰值带宽: 272.0 GB/s
- FP16 Tensor 峰值: 58.2 TFLOPS
- 拐点算术强度(fp16): 214.0 FLOPs/Byte
- PyTorch: 2.11.0+cu128   CUDA: 12.8
- Python: 3.11.16
- 锁频: 否（WSL2 不支持 nvidia-smi -lgc）
- 计时: CUDA event，每项 flush L2，重复取中位数
- 参数: bs=1 seq=2048 hidden=4096 heads=8 head_dim=64 dtype=fp16

## 基线性能

| 类别 | 算子 | shape | 访存(MB) | FLOPs(G) | 中位数(ms) | P99(ms) | 有效带宽(GB/s) | 带宽利用率(%) | 实测算力(TFLOPS) | 算力利用率(%) | Roofline 判定 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| elementwise / norm | residual_add | (1,2048,4096) | 48 | 0.008389 | 0.2314 | 0.8735 | 217.5 | 79.96 | 0.03625 | 0.06228 | memory-bound |
| elementwise / norm | gelu | (1,2048,4096) | 32 | 0.06711 | 0.1495 | 0.6912 | 224.4 | 82.51 | 0.4489 | 0.7713 | memory-bound |
| elementwise / norm | softmax(dim=-1) | (1,2048,4096) | 32 | 0.04194 | 0.1966 | 0.6799 | 170.7 | 62.75 | 0.2133 | 0.3666 | memory-bound |
| elementwise / norm | layer_norm | (1,2048,4096) | 32.02 | 0.06711 | 0.1546 | 0.6943 | 217.1 | 79.82 | 0.434 | 0.7457 | memory-bound |
| elementwise / norm | rms_norm (朴素实现) | (1,2048,4096) | 32.01 | 0.05033 | 1.378 | 2.11 | 24.36 | 8.958 | 0.03654 | 0.06278 | memory-bound |
| GEMM | matmul | (4096,4096)@(4096,4096) | 96 | 137.4 | 4.723 | 5.251 | 21.31 | 7.836 | 29.1 | 50 | compute-bound |
| attention | attn_naive (materialize) | B=1 H=8 S=2048 D=64 | 264 | 8.59 | 1.818 | 2.482 | 152.3 | 55.99 | 4.726 | 8.12 | memory-bound |
| attention | attn_sdpa (torch 融合) | B=1 H=8 S=2048 D=64 | 8 | 8.59 | 0.3389 | 0.8161 | 24.75 | 9.099 | 25.34 | 43.55 | compute-bound |

> 有效带宽 = 访存字节数 / 中位数延迟；利用率相对本卡峰值 272.0 GB/s 计算。
> 访存字节数按 `gpulab/workloads.py` 中的公式估算，假设已写入该文件。

## Attention 显存对比

| attention 实现 | 实测峰值显存(MB) | 理论 S/P 矩阵(MB) |
|---|---|---|
| attn_naive (materialize) | 496.1 | 128 |
| attn_sdpa (torch 融合) | 368.2 | 128 |

## 结论

（先跑出数字，再回来写结论。至少回答：）

1. 哪些算子是 memory-bound，哪些是 compute-bound？和 Roofline 预测一致吗？
2. attention 的 `attn_naive` 和 `attn_sdpa` 差距有多大？差在哪（延迟 / 显存）？
3. 你的 memory-bound 算子带宽利用率到了多少？离峰值还差多少？
4. 关闭 L2 冲刷（`--no-flush-l2`）重跑，带宽数字变成什么样？为什么？