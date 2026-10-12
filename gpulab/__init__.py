"""gpulab —— RTX 4060 Ti 算子优化项目的测量工具包。

模块划分：
    specs     显卡规格表 + Roofline 计算（峰值算力/带宽/拐点强度）
    timer     CUDA event 计时 harness（warmup / L2 flush / 分位数）
    workloads 各类算子的 FLOPs 与访存字节数公式
    verify    正确性校验（对比 PyTorch 参考实现）
    report    markdown 报告生成
"""

__version__ = "0.1.0"
