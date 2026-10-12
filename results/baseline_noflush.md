# PyTorch 原生算子基线（W1）

> 生成时间：2026-10-11 22:49:19

## 环境

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU (8GB, Ada, SM 8.9, 24 SM)
- 峰值带宽: 272.0 GB/s
- FP16 Tensor 峰值: 65.2 TFLOPS
- 拐点算术强度(fp16): 239.7 FLOPs/Byte
- PyTorch: 2.11.0+cu128   CUDA: 12.8
- Python: 3.11.16
- 锁频: 否（WSL2 不支持 nvidia-smi -lgc）
- 计时: CUDA event，每项 flush L2，重复取中位数
- 参数: bs=1 seq=2048 hidden=4096 heads=8 head_dim=64 dtype=fp16

## 基线性能

| 类别 | 算子 | shape | 访存(MB) | FLOPs(G) | 中位数(ms) | P99(ms) | 有效带宽(GB/s) | 带宽利用率(%) | 实测算力(TFLOPS) | 算力利用率(%) | Roofline 判定 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| elementwise / norm | residual_add | (1,2048,4096) | 48 | 0.008389 | 0.2089 | 0.2847 | 240.9 | 88.58 | 0.04016 | 0.06159 | memory-bound |
| elementwise / norm | gelu | (1,2048,4096) | 32 | 0.06711 | 0.06554 | 0.1536 | 512 | 188.2 | 1.024 | 1.571 | memory-bound |
| elementwise / norm | softmax(dim=-1) | (1,2048,4096) | 32 | 0.04194 | 0.2025 | 0.2621 | 165.7 | 60.92 | 0.2071 | 0.3177 | memory-bound |
| elementwise / norm | layer_norm | (1,2048,4096) | 32.02 | 0.06711 | 0.07885 | 0.1649 | 425.8 | 156.5 | 0.8511 | 1.305 | memory-bound |
| elementwise / norm | rms_norm (朴素实现) | (1,2048,4096) | 32.01 | 0.05033 | 1.314 | 2.057 | 25.55 | 9.392 | 0.03831 | 0.05876 | memory-bound |
| GEMM | matmul | (4096,4096)@(4096,4096) | 96 | 137.4 | 4.913 | 5.426 | 20.49 | 7.533 | 27.97 | 42.9 | compute-bound |
| attention | attn_naive (materialize) | B=1 H=8 S=2048 D=64 | 264 | 8.59 | 2.046 | 2.521 | 135.3 | 49.74 | 4.199 | 6.439 | memory-bound |
| attention | attn_sdpa (torch 融合) | B=1 H=8 S=2048 D=64 | 8 | 8.59 | 0.3512 | 0.425 | 23.88 | 8.781 | 24.46 | 37.51 | compute-bound |

> 有效带宽 = 访存字节数 / 中位数延迟；利用率相对本卡峰值 272.0 GB/s 计算。
> 访存字节数按 `gpulab/workloads.py` 中的公式估算，假设已写入该文件。

## Attention 显存对比

| attention 实现 | 实测峰值显存(MB) | 理论 S/P 矩阵(MB) |
|---|---|---|
| attn_naive (materialize) | 240.1 | 128 |
| attn_sdpa (torch 融合) | 112.2 | 128 |

## 结论

（先跑出数字，再回来写结论。至少回答：）

1. 哪些算子是 memory-bound，哪些是 compute-bound？和 Roofline 预测一致吗？
2. attention 的 `attn_naive` 和 `attn_sdpa` 差距有多大？差在哪（延迟 / 显存）？
3. 你的 memory-bound 算子带宽利用率到了多少？离峰值还差多少？
4. 关闭 L2 冲刷（`--no-flush-l2`）重跑，带宽数字变成什么样？为什么？