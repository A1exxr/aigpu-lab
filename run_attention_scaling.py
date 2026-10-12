"""W4 交付物生成器：attention 的「动机」数据。

扫描序列长度，对比标准 attention 与 torch SDPA 的**延迟**和**峰值显存**，
并**优雅处理 OOM** —— 在 8GB 卡上 "标准实现在 S=8192 直接跑不了" 本身就是
整个项目最有说服力的数据点。

用法
----
    python run_attention_scaling.py
    python run_attention_scaling.py --sweep 512,1024,2048,4096,8192
    python run_attention_scaling.py --no-flush-l2

    # W5 写完自己的 FlashAttention 后，直接挂进来对比，不用改这个脚本：
    python run_attention_scaling.py --extra "my_flash=kernels.w5_flash_attn:flash_attn_forward"

产出
----
    results/attention_scaling.md      markdown 报告（可直接进 README）
    results/attention_scaling.csv     原始数据
    results/fig_attention_latency.png 延迟 vs 序列长度
    results/fig_attention_memory.png  显存 vs 序列长度（带 OOM 标记）
    results/fig_roofline_attention.png Roofline 图
"""
from __future__ import annotations

import argparse
import csv
import importlib
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from gpulab import plotting, report, specs, timer, workloads

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def is_oom(e: BaseException) -> bool:
    return "out of memory" in str(e).lower() or type(e).__name__ == "OutOfMemoryError"


def peak_mem_mb(fn) -> float:
    """测一次调用的峰值显存（MB）。

    先释放计时用的 L2 冲刷缓冲区，否则那 256MB 会混进结果。
    """
    timer.release_flush_buffers()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024 ** 2


def load_extra(spec_str: str):
    """解析 'name=module:func'，把用户自己写的实现挂进来。"""
    if "=" not in spec_str or ":" not in spec_str:
        raise ValueError(f"--extra 格式应为 name=module:func，收到 {spec_str!r}")
    name, target = spec_str.split("=", 1)
    mod_name, fn_name = target.split(":", 1)
    mod = importlib.import_module(mod_name)
    return name.strip(), getattr(mod, fn_name)


def measure(fn, warmup: int, iters: int, flush_l2: bool):
    """返回 (stats, peak_mem_mb, err)。任何失败都不抛异常，只记 err。"""
    mem = None
    try:
        mem = peak_mem_mb(fn)
    except Exception as e:  # noqa: BLE001
        torch.cuda.empty_cache()
        return None, None, ("OOM" if is_oom(e) else f"{type(e).__name__}: {e}")

    try:
        stats = timer.bench(fn, warmup=warmup, iters=iters, flush_l2=flush_l2)
    except Exception as e:  # noqa: BLE001
        torch.cuda.empty_cache()
        return None, mem, ("OOM(bench)" if is_oom(e) else f"{type(e).__name__}: {e}")

    torch.cuda.empty_cache()
    return stats, mem, None


def main() -> int:
    ap = argparse.ArgumentParser(description="attention 序列长度扫描（W4 动机数据）")
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--sweep", default="512,1024,2048,4096,8192",
                    help="逗号分隔的序列长度列表")
    ap.add_argument("--dtype", default="fp16", choices=list(DTYPES))
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--no-flush-l2", action="store_true")
    ap.add_argument("--extra", action="append", default=None,
                    help="额外的实现，格式 name=module:func（可重复）")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    dtype = DTYPES[args.dtype]
    spec = specs.detect()
    seqs = [int(s) for s in args.sweep.split(",") if s.strip()]
    flush = not args.no_flush_l2
    B, H, D = args.bs, args.heads, args.head_dim
    scale = 1.0 / math.sqrt(D)

    extras = [load_extra(e) for e in (args.extra or [])]

    print("=" * 78)
    print(spec)
    print("=" * 78)
    print(f"B={B} H={H} D={D} dtype={args.dtype}  warmup={args.warmup} "
          f"iters={args.iters} flush_l2={flush}")
    print("=" * 78)

    rows = []          # 每个 (S, impl) 一条
    lat_series = {}    # 画图用
    mem_series = {}

    for S in seqs:
        print(f"\n--- S = {S} ---")
        try:
            q = torch.randn(B, H, S, D, device="cuda", dtype=dtype)
            k = torch.randn(B, H, S, D, device="cuda", dtype=dtype)
            v = torch.randn(B, H, S, D, device="cuda", dtype=dtype)
        except Exception as e:  # noqa: BLE001
            print(f"  分配 Q/K/V 就失败了（{e}），整个 S 记为 OOM")
            for name in ["attn_naive", "attn_sdpa"] + [n for n, _ in extras]:
                rows.append(dict(S=S, impl=name, ms=None, p99=None, mem=None,
                                 theory_mb=workloads.attention_naive_nbytes(
                                     B, H, S, D, dtype) / 1024 ** 2,
                                 bw=None, bw_util=None, tflops=None, bound="-",
                                 err="OOM"))
                lat_series.setdefault(name, []).append((S, None))
                mem_series.setdefault(name, []).append((S, None))
            q = k = v = None
            torch.cuda.empty_cache()
            continue

        def naive(q=q, k=k, v=v, scale=scale):
            scores = (q @ k.transpose(-2, -1)) * scale
            probs = F.softmax(scores, dim=-1)
            return probs @ v

        impls = [
            # (名字, 调用体, 是否会把 S/P 中间矩阵落到 HBM)
            ("attn_naive", naive, True),
            ("attn_sdpa",
             lambda q=q, k=k, v=v: F.scaled_dot_product_attention(q, k, v), False),
        ]
        for name, fn in extras:
            # 你自己写的实现一定是融合版，所以 materialize=False
            impls.append((name, (lambda fn=fn, q=q, k=k, v=v: fn(q, k, v)), False))

        for name, fn, materialize in impls:
            flops, nbytes = workloads.attention_core(
                B, H, S, D, dtype, materialize_scores=materialize)
            stats, mem, err = measure(fn, args.warmup, args.iters, flush)

            if stats:
                ms = stats["median_ms"]
                bw = workloads.bandwidth_gbps(nbytes, ms)
                tf = workloads.achieved_tflops(flops, ms)
                bound = spec.classify(flops, nbytes, args.dtype)
                print(f"  {name:<26} {ms:8.4f} ms   p99 {stats['p99_ms']:8.4f}   "
                      f"mem {mem:8.1f} MB   bw {bw:7.1f} GB/s ({100*bw/spec.bw_gbps:5.1f}%)"
                      f"   {bound}")
            else:
                ms = bw = tf = None
                bound = "-"
                print(f"  {name:<26} 失败: {err}")

            rows.append(dict(
                S=S, impl=name,
                ms=ms, p99=stats["p99_ms"] if stats else None, mem=mem,
                theory_mb=workloads.attention_naive_nbytes(B, H, S, D, dtype) / 1024 ** 2,
                bw=bw, bw_util=(100 * bw / spec.bw_gbps) if bw else None,
                tflops=tf, bound=bound, err=err or "",
            ))
            lat_series.setdefault(name, []).append((S, ms))
            mem_series.setdefault(name, []).append((S, mem))

        # 关键：上面那些闭包通过默认参数持有着 q/k/v 的引用，
        # 只 del 局部变量名并不能释放显存。必须先清掉闭包引用，
        # 否则下一个 S 迭代会在旧显存尚未释放时新建张量 → **假 OOM**，整张表就废了。
        impls = []
        fn = None
        naive = None
        torch.cuda.empty_cache()
        del q, k, v
        torch.cuda.empty_cache()

    # ---------------- 报告 ----------------
    if not rows:
        print("没有任何结果（--sweep 是空的？），退出。", file=sys.stderr)
        return 1

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with (out / "attention_scaling.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    main_rows = [[r["S"], r["impl"], r["ms"], r["p99"], r["mem"], r["bw"],
                  r["bw_util"], r["tflops"], r["bound"], r["err"]]
                 for r in rows]
    table = report.md_table(
        ["S", "实现", "中位延迟(ms)", "P99(ms)", "峰值显存(MB)", "有效带宽(GB/s)",
         "带宽利用率(%)", "实测算力(TFLOPS)", "Roofline", "备注"], main_rows)

    # 动机分析：用中等序列长度算算术强度
    mid = seqs[len(seqs) // 2]
    f_n, b_n = workloads.attention_core(B, H, S=mid, D=D, dtype=dtype,
                                        materialize_scores=True)
    f_f, b_f = workloads.attention_core(B, H, S=mid, D=D, dtype=dtype,
                                        materialize_scores=False)
    ai_n = f_n / b_n
    ai_f = f_f / b_f
    t_mem_n = b_n / (spec.bw_gbps * 1e9) * 1e3
    t_cmp_n = f_n / (spec.peak_tflops(args.dtype) * 1e12) * 1e3
    t_mem_f = b_f / (spec.bw_gbps * 1e9) * 1e3
    t_cmp_f = f_f / (spec.peak_tflops(args.dtype) * 1e12) * 1e3

    sections = [
        report.env_block(spec, extra=f"参数: B={B} H={H} D={D} sweep={seqs} "
                                     f"dtype={args.dtype}"),
        "## 扫描结果\n", table, "",
        "> 数值为 `None` 表示该配置失败（8GB 显存下大序列的标准实现会 OOM）。\n"
        "> 访存字节数按 `gpulab/workloads.py` 的公式估算，标准实现计入 S/P 两个\n"
        "> 中间矩阵的写+读（4×B·H·S² 个元素），融合实现不计入。\n",
        "## 为什么需要 FlashAttention（S = %d 的定量分析）\n" % mid,
        report.md_table(
            ["实现", "FLOPs(G)", "HBM 访存(MB)", "算术强度", "Roofline 判定",
             "访存耗时(ms)", "算力耗时(ms)", "理论上限(ms)"],
            [["标准 attention", f_n / 1e9, b_n / 1024 ** 2, ai_n,
              spec.classify(f_n, b_n, args.dtype), t_mem_n, t_cmp_n,
              max(t_mem_n, t_cmp_n)],
             ["FlashAttention 式融合", f_f / 1e9, b_f / 1024 ** 2, ai_f,
              spec.classify(f_f, b_f, args.dtype), t_mem_f, t_cmp_f,
              max(t_mem_f, t_cmp_f)]]), "",
        f"- 两者 **FLOPs 完全相同**：FlashAttention 不减少计算，只减少 **HBM 流量**。\n"
        f"- 访存从 {b_n/1024**2:.0f} MB 降到 {b_f/1024**2:.0f} MB（**{b_n/b_f:.0f}×**），"
        f"算术强度从 {ai_n:.1f} 升到 {ai_f:.0f}，"
        f"**瓶颈从带宽转移到算力**（本卡拐点 {spec.knee_intensity(args.dtype):.0f} FLOPs/Byte）。\n"
        f"- 理论上限从 {max(t_mem_n, t_cmp_n):.2f} ms 降到 {max(t_mem_f, t_cmp_f):.2f} ms"
        f"（约 {max(t_mem_n, t_cmp_n)/max(t_mem_f, t_cmp_f):.1f}×）。\n"
        f"- ⚠️ **但实际拿不到这个倍数**：真实 kernel 的 Tensor Core 利用率通常只有\n"
        f"  30%–60%，还有 SRAM 访存、重算、边界处理开销。报告里要写清\n"
        f"  「理论上限 vs 实际达成」的差距。\n",
        "## 显存：O(S²) vs O(S)\n",
        report.md_table(["S", "理论 S/P 矩阵(MB)"],
                        [[s, workloads.attention_naive_nbytes(B, H, s, D, dtype) / 1024 ** 2]
                         for s in seqs]), "",
        "## 结论（自己填）\n",
        "1. 标准 attention 在哪个 S 开始 OOM？和上表理论值对得上吗？",
        "2. SDPA 相对标准实现的加速比随 S 怎么变化？为什么？",
        "3. 实测带宽利用率到多少？离峰值 272 GB/s 还差多少？",
        "4. 你的实现在哪个 S 上开始赢过 SDPA？",
    ]
    report.write_report(out / "attention_scaling.md",
                        "Attention 序列长度扫描（W4 动机数据）", sections)

    # ---------------- 画图 ----------------
    plotting.plot_scaling(lat_series, out / "fig_attention_latency.png",
                          ylabel="Latency (ms)",
                          title="Attention latency vs sequence length")
    plotting.plot_memory(mem_series, out / "fig_attention_memory.png")

    # Roofline：把每个实现最大的那个点画上去
    points = []
    for name in lat_series:
        cand = [r for r in rows if r["impl"] == name and r["tflops"]]
        if not cand:
            continue
        r = max(cand, key=lambda x: x["S"])
        materialize = "naive" in name
        fl, nb = workloads.attention_core(B, H, r["S"], D, dtype,
                                          materialize_scores=materialize)
        points.append(dict(name=f"{name} (S={r['S']})", ai=fl / nb,
                           tflops=r["tflops"]))
    if points:
        plotting.plot_roofline(spec, points, out / "fig_roofline_attention.png",
                               dtype=args.dtype,
                               title=f"Attention on {spec.name}")

    print(f"\n全部产出在 {out.resolve()}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
