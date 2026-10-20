# 自写 CUDA Kernel 性能报告

> 生成时间：2026-10-16 19:02:35

## 环境

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU (8GB, Ada, SM 8.9, 24 SM)
- 峰值带宽: 272.0 GB/s
- FP16 Tensor 峰值: 65.2 TFLOPS
- 拐点算术强度(fp16): 239.7 FLOPs/Byte
- PyTorch: 2.11.0+cu128   CUDA: 12.8
- Python: 3.11.16
- 锁频: 否（WSL2 不支持 nvidia-smi -lgc）
- 计时: CUDA event，每项 flush L2，重复取中位数

## 结果

| kernel | case | 状态 | 我的(ms) | PyTorch(ms) | 加速比 | 峰值显存(MB) | 带宽利用率(%) | Roofline | 备注 |
|---|---|---|---|---|---|---|---|---|---|
| softmax | (1,2048,4096) fp16 | PASS | 0.216 | 0.1976 | 0.9151 | 64 | 57.12 | memory-bound |  |

> 状态含义：`PASS` 正确且已测；`WRONG` 结果不对（先修正确性再谈性能）；
> `SKIP` 文件还没写；`FAIL` 调用/编译出错。

## W2/W3 验收标准

- W2：至少一个算子**带宽利用率 > 70%**（即 > 190 GB/s），且最大相对误差 < 1e-3
- W3：融合算子相对分步实现有明确加速比（memory-bound 场景通常 1.5x–3x），
  并能算出节省了多少次 HBM 往返

## 每个算子都要能回答的问题

1. `block_size` 为什么是这个值？改成 128 或 512 性能怎么变？（**测出来**）
2. 用了什么规约策略？warp shuffle / 共享内存 / 树形规约？
3. 共享内存用了多少字节？occupancy 是多少？
4. 有效带宽到了峰值的百分之多少？
5. 有没有 bank conflict / warp divergence？怎么验证的？
