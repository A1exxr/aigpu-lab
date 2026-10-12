"""CUDA-event 计时工具。项目里所有性能数字都必须经由这里产生。"""
from __future__ import annotations

import statistics

import torch

# 用于刷 L2 的大缓冲区（按 device 缓存，避免每次重新分配）
_L2_FLUSH_BUFFERS: dict = {}


def _flush_l2(device, mb: int = 256) -> None:
    """向一块远大于 L2 的缓冲区写入，把上一轮的数据从 L2 里挤出去。

    为什么必须做这件事：
        如果连续测量同一个算子、输入张量还是同一块内存，那么第二次之后
        输入很可能命中 L2，你测到的就不是 HBM 带宽了 —— 有效带宽会虚高，
        甚至算出超过显存理论带宽的荒谬结果。
        真实部署时数据来自 HBM，所以测 HBM 性能必须先把 L2 冲掉。

    注意：这次写入发生在 start.record() 之前，不计入被测时间。
    """
    buf = _L2_FLUSH_BUFFERS.get(device)
    if buf is None or buf.numel() < mb * 1024 * 1024 // 4:
        buf = torch.empty(mb * 1024 * 1024 // 4, dtype=torch.float32, device=device)
        _L2_FLUSH_BUFFERS[device] = buf
    buf.zero_()


def release_flush_buffers() -> None:
    """释放 L2 冲刷缓冲区。

    **测显存占用前必须调用。** 那个缓冲区有 256MB，会被算进
    ``torch.cuda.max_memory_allocated()``，让你的「峰值显存」虚高 256MB，
    在小显存卡上足以改变结论。
    """
    _L2_FLUSH_BUFFERS.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def bench(fn, warmup: int = 50, iters: int = 200, flush_l2: bool = True) -> dict:
    """对 ``fn()`` 做预热 + 多次计时，返回毫秒级统计量。

    参数
    ----
    fn : callable
        无参可调用对象，内部完成一次"被测操作"。
    warmup : int
        预热次数，结果丢弃。排除 JIT 编译、autotune、cache 冷启动、
        GPU 频率爬升带来的偏差。
    iters : int
        正式计时次数。建议 >= 200。
    flush_l2 : bool
        每次计时前是否冲刷 L2（见 ``_flush_l2``）。测显存带宽类算子
        必须为 True；测纯计算（如大 GEMM）影响较小。

    返回
    ----
    dict，含 median_ms / mean_ms / min_ms / p99_ms / std_ms。
    **报告时用 median_ms，不要用 mean_ms** —— 长尾会污染平均值。
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "torch.cuda.is_available() 为 False。请先按《环境搭建清单》排查 WSL2 + 驱动。"
        )

    device = torch.cuda.current_device()

    # ---- 预热 ----
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # ---- 正式计时 ----
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times: list[float] = []

    for _ in range(iters):
        if flush_l2:
            _flush_l2(device)
        start.record()
        fn()
        end.record()
        # 每轮末尾同步一次：CUDA 是异步的，不同步的话 end 记录的是
        # "命令入队完成"的时间，而不是"GPU 算完"的时间。
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    n = len(times)
    return {
        "iters": n,
        "median_ms": times[n // 2],
        "mean_ms": statistics.fmean(times),
        "min_ms": times[0],
        "p99_ms": times[min(n - 1, int(round(n * 0.99)))],
        "std_ms": statistics.pstdev(times) if n > 1 else 0.0,
    }


def bench_report(name: str, fn, **kwargs) -> dict:
    """``bench`` 的便捷包装：跑完顺便打印一行人类可读的结果。"""
    stats = bench(fn, **kwargs)
    print(
        f"{name:<28} median={stats['median_ms']:8.4f} ms  "
        f"p99={stats['p99_ms']:8.4f} ms  min={stats['min_ms']:8.4f} ms"
    )
    return stats
