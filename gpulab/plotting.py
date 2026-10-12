"""画图工具：Roofline 图 + 序列长度伸缩曲线。

matplotlib 是**可选依赖**：没装时只跳过画图，不影响测量与报告生成。

设计说明
--------
图里统一用**英文标签**。原因：WSL 里通常没有中文字体，用中文会渲染成一堆方框（tofu）。
英文标签在任何环境下都不会出错，而且技术报告里本来就是通用做法。
如果你的系统装了中文字体，可以用 --cjk-font 指定。
"""
from __future__ import annotations

from pathlib import Path


def _mpl(cjk_font: str | None = None):
    """惰性导入 matplotlib。没装就返回 None，让调用方优雅跳过。"""
    try:
        import matplotlib
        matplotlib.use("Agg")  # 无显示环境（WSL）必须用 Agg
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    if cjk_font:
        plt.rcParams["font.sans-serif"] = [cjk_font]
        plt.rcParams["axes.unicode_minus"] = False
    return plt


def plot_roofline(spec, points, out_path, dtype: str = "fp16",
                  title: str | None = None, cjk_font: str | None = None) -> bool:
    """画 Roofline 图。

    参数
    ----
    spec   : gpulab.specs.GpuSpec
    points : list[dict]，每项含
             name(str), ai(float, FLOPs/Byte), tflops(float 实测),
             可选 marker / color
    out_path : 输出 PNG 路径

    返回是否真的画了图。
    """
    plt = _mpl(cjk_font)
    if plt is None:
        print("[plotting] 未安装 matplotlib，跳过 Roofline 图。"
              "装它：pip install matplotlib")
        return False

    peak = spec.peak_tflops(dtype)
    bw = spec.bw_gbps
    knee = spec.knee_intensity(dtype)

    fig, ax = plt.subplots(figsize=(9, 6))

    # --- Roofline 天花板：两条约束线 + 取小 ---
    import numpy as np
    xs = np.logspace(-1, 5, 400)
    mem_line = xs * bw / 1000.0          # GB/s * FLOPs/Byte -> GFLOPs/s -> TFLOPS
    roof = np.minimum(mem_line, peak)

    ax.plot(xs, mem_line, "--", color="#888888", lw=1.2,
            label=f"Memory roofline ({bw:.0f} GB/s)")
    ax.axhline(peak, ls=":", color="#888888", lw=1.2,
               label=f"Compute peak ({peak:.1f} TFLOPS, {dtype})")
    ax.plot(xs, roof, "-", color="#1f3b73", lw=2.0, label="Attainable")
    ax.axvline(knee, ls="-.", color="#c00000", lw=1.0,
               label=f"Knee = {knee:.0f} FLOPs/Byte")

    # --- 实测点 ---
    for p in points:
        ax.scatter(p["ai"], p["tflops"], s=90, zorder=5,
                   marker=p.get("marker", "o"),
                   color=p.get("color", "#e07b00"),
                   edgecolor="black", linewidths=0.6)
        ax.annotate(p["name"], (p["ai"], p["tflops"]),
                    textcoords="offset points", xytext=(7, 6), fontsize=8)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic intensity (FLOPs / Byte)")
    ax.set_ylabel(f"Achieved performance (TFLOPS, {dtype})")
    ax.set_title(title or f"Roofline - {spec.name}")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plotting] Roofline 图已写入 {out_path}")
    return True


def plot_scaling(series, out_path, xlabel: str = "Sequence length",
                 ylabel: str = "Latency (ms)", title: str = "",
                 logy: bool = True, cjk_font: str | None = None) -> bool:
    """画多条曲线（例如不同实现随序列长度的变化）。

    参数
    ----
    series : dict[str, list[tuple[x, y]]]
             例：{"attn_naive": [(512, 0.31), (1024, 1.02)], "attn_sdpa": [...]}
             y 为 None 的点会被跳过（OOM 就是这种情况）
    """
    plt = _mpl(cjk_font)
    if plt is None:
        print("[plotting] 未安装 matplotlib，跳过曲线图。"
              "装它：pip install matplotlib")
        return False

    fig, ax = plt.subplots(figsize=(9, 6))
    for name, pts in series.items():
        pts = [(x, y) for x, y in pts if y is not None]
        if not pts:
            continue
        xs, ys = zip(*sorted(pts))
        ax.plot(xs, ys, "o-", lw=1.8, ms=6, label=name)

    if logy:
        ax.set_yscale("log")
    ax.set_xscale("log", base=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=9)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plotting] 曲线图已写入 {out_path}")
    return True


def plot_memory(series, out_path, cjk_font: str | None = None) -> bool:
    """显存占用 vs 序列长度。OOM 的点会在最上方画一个叉 —— 这是最有说服力的图。"""
    plt = _mpl(cjk_font)
    if plt is None:
        print("[plotting] 未安装 matplotlib，跳过显存图。")
        return False

    fig, ax = plt.subplots(figsize=(9, 6))
    oom_xs = []
    for name, pts in series.items():
        good = [(x, y) for x, y in pts if y is not None]
        if good:
            xs, ys = zip(*sorted(good))
            ax.plot(xs, ys, "o-", lw=1.8, ms=6, label=name)
        oom = [x for x, y in pts if y is None]
        oom_xs.extend(oom)

    if oom_xs:
        ymax = ax.get_ylim()[1]
        ax.scatter(sorted(set(oom_xs)), [ymax] * len(set(oom_xs)),
                   marker="x", s=110, color="#c00000", zorder=6,
                   label="OOM (out of memory)")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Peak GPU memory (MB)")
    ax.set_title("Attention memory scaling: O(S^2) vs O(S)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=9)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plotting] 显存图已写入 {out_path}")
    return True
