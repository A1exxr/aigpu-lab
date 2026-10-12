"""报告生成：把测量结果输出成 markdown，直接贴进 README / 博客。"""
from __future__ import annotations

import pathlib
from datetime import datetime


def md_table(headers: list[str], rows: list[list]) -> str:
    """生成 markdown 表格。"""
    def fmt(v):
        if v is None:
            return "—"
        if isinstance(v, float):
            if v != v:  # nan
                return "n/a"
            if v == float("inf"):
                return "inf"
            return f"{v:.4g}"
        return str(v)

    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(fmt(c) for c in r) + " |")
    return "\n".join(lines)


def env_block(gpu, extra: str = "") -> str:
    """生成报告顶部的环境信息块。"""
    import torch
    import sys
    lines = [
        "## 环境\n",
        f"- GPU: {gpu.name} ({gpu.vram_gb:g}GB, {gpu.arch}, SM {gpu.cc[0]}.{gpu.cc[1]}, "
        f"{gpu.sm} SM)",
        f"- 峰值带宽: {gpu.bw_gbps:.1f} GB/s",
        f"- FP16 Tensor 峰值: {gpu.fp16_tc_tflops} TFLOPS",
        f"- 拐点算术强度(fp16): {gpu.knee_intensity('fp16'):.1f} FLOPs/Byte",
        f"- PyTorch: {torch.__version__}   CUDA: {torch.version.cuda}",
        f"- Python: {sys.version.split()[0]}",
        "- 锁频: 否（WSL2 不支持 nvidia-smi -lgc）",
        "- 计时: CUDA event，每项 flush L2，重复取中位数",
    ]
    if extra:
        lines.append(f"- {extra}")
    return "\n".join(lines) + "\n"


def write_report(path: str | pathlib.Path, title: str, sections: list[str]) -> pathlib.Path:
    """把若干 section 拼成一份带时间戳的报告文件。"""
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    body = [f"# {title}\n",
            f"> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"]
    body.extend(sections)
    p.write_text("\n".join(body), encoding="utf-8")
    print(f"\n报告已写入: {p}")
    return p
