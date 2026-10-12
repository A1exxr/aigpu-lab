"""边跑负载边记录 GPU 时钟 / 功耗 / 温度，用来发现热降频与功耗墙。

为什么需要它
------------
笔记本 GPU 有两个桌面卡没有的问题：
  1. 功耗墙由厂商配置（你的机器 Default 55W / Current 126W / Max 140W），
     电源模式一变性能就可能腰斩；
  2. 持续满载会热降频，**benchmark 越跑越慢，数字会飘**。
这两个问题用肉眼看不出来，必须采样记录。

用法
----
    # 只监视 30 秒（你自己在另一个终端跑负载）
    python gpu_watch.py --duration 30

    # 包住一条命令：跑它的同时后台采样，结束后自动出报告（推荐）
    python gpu_watch.py -- python run_baseline.py

    # 跑你自己的 kernel 测试
    python gpu_watch.py --csv results/watch_rmsnorm.csv -- python kernels/bench_kernels.py

输出
----
  · results/gpu_watch.csv —— 逐次采样（可直接画图）
  · 终端报告：SM 时钟的中位/最大、功耗、温度、**降频漂移比**
"""
from __future__ import annotations

import argparse
import csv
import shutil
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

QUERY_FIELDS = [
    "clocks.sm", "clocks.mem", "power.draw", "temperature.gpu",
    "pstate", "utilization.gpu",
]
QUERY = ",".join(QUERY_FIELDS)


def _num(s: str):
    """nvidia-smi 在 WSL2 下常把传感器字段返回 N/A，需要容错。"""
    s = s.strip()
    if not s or s.upper() in ("N/A", "[N/A]", "[NOT SUPPORTED]", "NOT FOUND"):
        return None
    try:
        return float(s)
    except ValueError:
        return s


def sample() -> list:
    out = subprocess.run(
        ["nvidia-smi", f"--query-gpu={QUERY}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()
    row = [p.strip() for p in out[0].split(",")]
    return [_num(v) for v in row]


def _loop(samples, stop, interval, duration):
    t0 = time.time()
    while not stop.is_set():
        try:
            row = sample()
        except Exception as e:  # noqa: BLE001
            print(f"[gpu_watch] 采样失败: {e}", file=sys.stderr)
            break
        samples.append((time.time() - t0, row))
        if duration and (time.time() - t0) >= duration:
            break
        stop.wait(interval)


def report(samples, csv_path: Path) -> None:
    if not samples:
        print("\n[gpu_watch] 没采到任何样本。")
        return

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t_sec"] + QUERY_FIELDS)
        for t, row in samples:
            w.writerow([f"{t:.2f}"] + ["" if v is None else v for v in row])

    def col(i):
        return [r[i] for _, r in samples if isinstance(r[i], (int, float))]

    sm = col(0)
    mem = col(1)
    pw = col(2)
    temp = col(3)

    print(f"\n{'=' * 66}")
    print(f"GPU 采样报告（{len(samples)} 个样本，已写入 {csv_path}）")
    print(f"{'=' * 66}")

    if sm:
        print(f"SM 时钟   : 中位 {statistics.median(sm):7.0f} MHz   "
              f"最大 {max(sm):7.0f} MHz   最小 {min(sm):7.0f} MHz")
    else:
        print("SM 时钟   : 无数据（WSL2 传感器限制）")
    if mem:
        print(f"显存时钟  : 中位 {statistics.median(mem):7.0f} MHz   "
              f"最大 {max(mem):7.0f} MHz")
    if pw:
        print(f"功耗      : 中位 {statistics.median(pw):7.2f} W     "
              f"最大 {max(pw):7.2f} W")
    if temp:
        print(f"温度      : 中位 {statistics.median(temp):7.0f} °C    "
              f"最大 {max(temp):7.0f} °C")

    # 降频漂移：把样本四等分，比较首尾两段的 SM 时钟中位数
    if len(sm) >= 8:
        q = len(sm) // 4
        first = statistics.median(sm[:q])
        last = statistics.median(sm[-q:])
        ratio = last / first if first else 0
        print(f"\n降频漂移  : 前 25% 中位 {first:.0f} MHz → "
              f"后 25% 中位 {last:.0f} MHz   (比值 {ratio:.3f})")
        if ratio < 0.92:
            print("  ⚠️  明显热降频：长跑会让性能下降 "
                  f"{(1 - ratio) * 100:.1f}%。"
                  "\n     → benchmark 要短、组间要歇、对比实验必须背靠背跑完。")
        else:
            print("  ✅ 时钟稳定，没有明显热降频。")
    else:
        print("\n降频漂移  : 样本太少（<8），无法判断。建议 --duration 至少 30 秒。")

    print(f"{'=' * 66}")
    print("把这份报告的环境数据抄进你的 README：功耗墙、SM 时钟区间、温度上限。")
    print("面试官问「你的数字可复现吗」，这就是答案。")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="跑负载的同时记录 GPU 时钟/功耗/温度",
        epilog="用 `--` 把要包住的命令放最后，例如： python gpu_watch.py -- python run_baseline.py",
    )
    ap.add_argument("--duration", type=float, default=None,
                    help="监视多少秒（不包裹命令时用；默认一直跑到 Ctrl-C）")
    ap.add_argument("--interval", type=float, default=1.0, help="采样间隔秒数（默认 1.0）")
    ap.add_argument("--csv", default="results/gpu_watch.csv", help="CSV 输出路径")
    ap.add_argument("cmd", nargs=argparse.REMAINDER,
                    help="要包裹的命令（前面加 --）")
    args = ap.parse_args()

    if shutil.which("nvidia-smi") is None:
        print("找不到 nvidia-smi。确认你在 WSL2 里、且 Windows 侧驱动正常。",
              file=sys.stderr)
        return 2

    cmd = [a for a in args.cmd if a != "--"]
    samples: list = []
    stop = threading.Event()
    watcher = threading.Thread(
        target=_loop, args=(samples, stop, args.interval, args.duration),
        daemon=True,
    )

    print(f"[gpu_watch] 开始采样，间隔 {args.interval}s")
    watcher.start()

    rc = 0
    try:
        if cmd:
            print(f"[gpu_watch] 包裹命令: {' '.join(cmd)}")
            rc = subprocess.call(cmd)
        else:
            while watcher.is_alive():
                watcher.join(timeout=0.5)
    except KeyboardInterrupt:
        print("\n[gpu_watch] 收到 Ctrl-C，停止采样。")
    finally:
        stop.set()
        watcher.join(timeout=args.interval * 2 + 2)

    report(samples, Path(args.csv))
    return rc


if __name__ == "__main__":
    sys.exit(main())
