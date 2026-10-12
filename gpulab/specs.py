"""显卡规格表与 Roofline 计算。

设计意图：峰值算力和峰值带宽 **没法从 torch 查出来**，只能查厂商规格。
所以这里内置一张表，并强制你在跑实验前先确认自己的卡在上面。
"先查规格，再动手" 本身就是这个项目的第一个练习。
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import torch

# ---------------------------------------------------------------------------
# 规格表
#
# ⚠️ 算力口径必须统一，否则跨卡比较会得出错误结论。这里统一采用：
#
#     fp16_tc_tflops = 厂商标称的 FP16 Tensor Core 峰值（**FP16 累加**）
#     fp8_tc_tflops  = 厂商标称的 FP8  Tensor Core 峰值（**FP16 累加**）
#
#   两种口径的差别（很容易踩坑）：
#     - GeForce 卡（Ada / Ampere）：FP16 累加 = 2 × FP32 累加。
#       查 NVIDIA 产品页会看到两个数，本站取的是**大的那个**（标称值）。
#       4090 的 FP32 累加是 165.2，FP16 累加是 330.3。
#     - 数据中心卡（A100 / H100）：标称值本身就是 FP32 累加口径，
#       不存在 GeForce 那样的翻倍关系。稀疏（2:4）另外再翻倍，本站不计。
#
#   如果你要对比的算子里用的是 FP32 累加，把本表的值除以 2 再算利用率，
#   否则会把"利用率"算低一半。
#
#   数值以 NVIDIA 官方规格为准，这里给的是常见值 —— 请自行核对你的卡。
# ---------------------------------------------------------------------------
GPU_SPECS: dict[str, dict] = {
    # ---- Ada Lovelace (SM 8.9) ----
    # 笔记本卡：TGP 由厂商配置，同一型号不同笔记本的实际性能可差一倍以上。
    #
    # ★ 本条目已用**实测持续频率**标定（2026-10-11，gpu_watch.py 采样）：
    #     实测持续 SM 时钟 = 2655 MHz    （中位；功耗峰值 89W，温度峰值 66°C）
    #     厂商标称 boost   = 2370 MHz
    #     nvidia-smi 报告   Max Clocks = 3105 MHz  ← 硬件上限，永远跑不到
    #
    #   实测频率比标称**高 12%**，说明这台机器散热与功耗墙都够用、没撞墙。
    #   换机器必须重新跑 `python gpu_watch.py` 再改这里的 clock_ghz —— 否则
    #   所有「算力利用率」都是错的（而且错得隐蔽：数字看着正常，只是系统性偏移）。
    "RTX 4060 Laptop GPU": dict(
        arch="Ada", cc=(8, 9), sm=24, cuda_cores=3072, clock_ghz=2.655,
        bw_gbps=272.0, fp32_tflops=16.3, fp16_tc_tflops=65.2, fp8_tc_tflops=130.5,
        fp4_tc_tflops=None,  # Ada 不支持 FP4
        tgp_note="实测持续 2655 MHz / 峰值 89W / 66°C；功耗墙 126W（上限 140W，Default 仅 55W）",
    ),
    "RTX 4060 Ti": dict(
        arch="Ada", cc=(8, 9), sm=34, cuda_cores=4352, clock_ghz=2.535,
        bw_gbps=288.0, fp32_tflops=22.1, fp16_tc_tflops=88.3, fp8_tc_tflops=176.6,
        fp4_tc_tflops=None,  # Ada 不支持 FP4
    ),
    "RTX 4070 Ti Super": dict(
        arch="Ada", cc=(8, 9), sm=66, cuda_cores=8448, clock_ghz=2.61,
        bw_gbps=672.0, fp32_tflops=44.1, fp16_tc_tflops=176.4, fp8_tc_tflops=352.8,
        fp4_tc_tflops=None,
    ),
    "RTX 4080 Super": dict(
        arch="Ada", cc=(8, 9), sm=80, cuda_cores=10240, clock_ghz=2.55,
        bw_gbps=736.0, fp32_tflops=52.2, fp16_tc_tflops=208.9, fp8_tc_tflops=417.8,
        fp4_tc_tflops=None,
    ),
    "RTX 4090": dict(
        arch="Ada", cc=(8, 9), sm=128, cuda_cores=16384, clock_ghz=2.52,
        bw_gbps=1008.0, fp32_tflops=82.6, fp16_tc_tflops=330.3, fp8_tc_tflops=660.6,
        fp4_tc_tflops=None,
    ),
    # ---- Ampere (SM 8.6) ----
    "RTX 3090": dict(
        arch="Ampere", cc=(8, 6), sm=82, cuda_cores=10496, clock_ghz=1.695,
        bw_gbps=936.0, fp32_tflops=35.6, fp16_tc_tflops=142.3, fp8_tc_tflops=None,
        fp4_tc_tflops=None,  # Ampere 不支持 FP8
    ),
    "RTX 3080": dict(
        arch="Ampere", cc=(8, 6), sm=68, cuda_cores=8704, clock_ghz=1.71,
        bw_gbps=760.0, fp32_tflops=29.8, fp16_tc_tflops=119.2, fp8_tc_tflops=None,
        fp4_tc_tflops=None,
    ),
    # ---- Hopper / Blackwell（数据中心卡，供参考）----
    "A100 SXM 80GB": dict(
        arch="Ampere", cc=(8, 0), sm=108, cuda_cores=6912, clock_ghz=1.41,
        bw_gbps=2039.0, fp32_tflops=19.5, fp16_tc_tflops=312.0, fp8_tc_tflops=None,
        fp4_tc_tflops=None,
    ),
    "H100 SXM": dict(
        arch="Hopper", cc=(9, 0), sm=132, cuda_cores=16896, clock_ghz=1.98,
        bw_gbps=3350.0, fp32_tflops=66.9, fp16_tc_tflops=989.4, fp8_tc_tflops=1978.9,
        fp4_tc_tflops=None,
    ),
}


@dataclass
class GpuSpec:
    name: str
    arch: str
    cc: tuple[int, int]
    sm: int
    cuda_cores: int
    clock_ghz: float
    bw_gbps: float
    fp32_tflops: float
    fp16_tc_tflops: float | None
    fp8_tc_tflops: float | None
    fp4_tc_tflops: float | None
    vram_gb: float
    tgp_note: str = ""

    # ---- Roofline 相关 ----

    def peak_tflops(self, dtype: str) -> float:
        """按 dtype 返回峰值算力（TFLOPS）。"""
        table = {
            "fp32": self.fp32_tflops,
            "fp16": self.fp16_tc_tflops or self.fp32_tflops,
            "bf16": self.fp16_tc_tflops or self.fp32_tflops,
            "fp8": self.fp8_tc_tflops,
            "fp4": self.fp4_tc_tflops,
        }
        if dtype not in table:
            raise ValueError(f"未知 dtype: {dtype}，可选 {list(table)}")
        val = table[dtype]
        if val is None:
            raise ValueError(
                f"{self.name}（{self.arch}）不支持 {dtype} 硬件路径。"
                f"如果你以为自己在做 {dtype} 实验，那多半是软件模拟（反量化），性能会更差。"
            )
        return val

    def knee_intensity(self, dtype: str = "fp16") -> float:
        """拐点算术强度（FLOPs/Byte）。

        算术强度低于这个值的算子在这张卡上是 memory-bound。
        值越大 = 这张卡越"缺带宽"，memory-bound 优化越重要。
        """
        return self.peak_tflops(dtype) * 1e12 / (self.bw_gbps * 1e9)

    def classify(self, flops: float, nbytes: float, dtype: str = "fp16") -> str:
        """判断给定工作负载是 compute-bound 还是 memory-bound。"""
        ai = flops / nbytes if nbytes > 0 else float("inf")
        return "compute-bound" if ai > self.knee_intensity(dtype) else "memory-bound"

    def attainable_tflops(self, flops: float, nbytes: float, dtype: str = "fp16") -> float:
        """Roofline 上该工作负载的理论性能上限（TFLOPS）。"""
        ai = flops / nbytes if nbytes > 0 else float("inf")
        return min(ai * self.bw_gbps, self.peak_tflops(dtype))

    def with_measured_clock(self, clock_ghz: float) -> "GpuSpec":
        """用**实测持续 SM 时钟**重算峰值算力。

        为什么必须这么做：
            厂商标称的 boost 频率是「最好情况下能摸到」，笔记本卡的实际持续频率
            由功耗墙与散热决定，往往跑不到标称值。
            `nvidia-smi` 里的 `clocks.max.sm`（例如 3105 MHz）是**硬件上限**，更不能用。
            Roofline 分析里的「算力利用率」如果用了错频率，整个结论就是错的。

        公式（Ada / Ampere GeForce，FP16 累加口径）：
            FP32        = CUDA核心数 × 2 × 频率(GHz) / 1000        [TFLOPS]
            FP16 Tensor = FP32 × 4
            FP8  Tensor = FP16 × 2（仅 Ada 及以上支持）

        用法：
            python gpu_watch.py -- python run_baseline.py   # 拿到持续 SM 时钟
            >>> s = specs.detect().with_measured_clock(2.60)
        """
        if clock_ghz <= 0:
            raise ValueError("时钟必须为正数")
        fp32 = self.cuda_cores * 2 * clock_ghz / 1000.0
        fp16 = fp32 * 4
        fp8 = fp16 * 2 if self.fp8_tc_tflops is not None else None
        return replace(
            self,
            clock_ghz=clock_ghz,
            fp32_tflops=round(fp32, 1),
            fp16_tc_tflops=round(fp16, 1),
            fp8_tc_tflops=round(fp8, 1) if fp8 is not None else None,
        )

    def __str__(self) -> str:
        note = f"\n  备注          : {self.tgp_note}" if self.tgp_note else ""
        return (
            f"{self.name}  [{self.arch}, SM {self.cc[0]}.{self.cc[1]}, {self.sm} SM, "
            f"{self.vram_gb:g}GB]\n"
            f"  峰值带宽      : {self.bw_gbps:>8.1f} GB/s\n"
            f"  FP32          : {self.fp32_tflops:>8.1f} TFLOPS\n"
            f"  FP16 Tensor   : "
            f"{self.fp16_tc_tflops if self.fp16_tc_tflops else 'N/A':>8} TFLOPS\n"
            f"  FP8  Tensor   : "
            f"{self.fp8_tc_tflops if self.fp8_tc_tflops else '不支持':>8}\n"
            f"  FP4  Tensor   : "
            f"{self.fp4_tc_tflops if self.fp4_tc_tflops else '不支持'}\n"
            f"  拐点算术强度  : {self.knee_intensity('fp16'):>8.1f} FLOPs/Byte (fp16)"
            f"{note}"
        )


def detect(name_filter: str | None = None) -> GpuSpec:
    """探测当前 GPU，从规格表里匹配出峰值参数。

    匹配不上时会报错并提示你手动补充规格 —— 这是刻意的：
    没有峰值参数，后面所有"利用率"都没法算。
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，无法探测 GPU。")

    raw = torch.cuda.get_device_name(0)
    cc = torch.cuda.get_device_capability(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3

    query = name_filter or raw
    for key, spec in GPU_SPECS.items():
        if key.lower() in query.lower():
            return GpuSpec(name=raw, vram_gb=round(vram_gb, 1), **spec)

    raise KeyError(
        f"规格表里没有匹配 '{raw}'（计算能力 SM {cc[0]}.{cc[1]}，显存 {vram_gb:.1f}GB）。\n"
        f"请到 NVIDIA 官网查到它的峰值带宽、FP16/FP8 Tensor Core 吞吐，"
        f"补进 gpulab/specs.py 的 GPU_SPECS。\n"
        f"已有的条目：{', '.join(GPU_SPECS)}"
    )


if __name__ == "__main__":
    s = detect()
    print(s)
    print()
    for dt in ("fp16", "fp8"):
        try:
            print(f"  {dt} 拐点强度 = {s.knee_intensity(dt):.1f} FLOPs/Byte")
        except ValueError as e:
            print(f"  {dt}: {e}")
