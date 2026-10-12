"""环境自检：一条命令告诉你还差什么、怎么补。

用法：
    python check_env.py

它会逐项检查 Python / PyTorch / CUDA / 显卡识别 / Triton / CUDA Toolkit /
Nsight 工具链，把失败项的**修复命令**直接打印出来。
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

RESULTS: list[tuple[str, bool, str, str]] = []


def check(name: str, fn, fix: str) -> bool:
    try:
        ok, detail = fn()
    except Exception as e:  # noqa: BLE001
        ok, detail = False, f"{type(e).__name__}: {e}"
    RESULTS.append((name, bool(ok), str(detail), fix if not ok else ""))
    return bool(ok)


# ---------------------------------------------------------------------------
# 各项检查
# ---------------------------------------------------------------------------

def c_python():
    v = sys.version_info
    return v >= (3, 9), f"{v.major}.{v.minor}.{v.micro}  ({sys.executable})"


def c_torch():
    if importlib.util.find_spec("torch") is None:
        return False, "未安装"
    import torch
    return True, torch.__version__


def c_cuda():
    if importlib.util.find_spec("torch") is None:
        return False, "torch 未安装，无法判断"
    import torch
    return torch.cuda.is_available(), (
        f"torch.version.cuda = {torch.version.cuda}"
        if torch.cuda.is_available()
        else "torch.cuda.is_available() = False"
    )


def c_device():
    if importlib.util.find_spec("torch") is None:
        return False, "torch 未安装"
    import torch
    if not torch.cuda.is_available():
        return False, "无可用 CUDA 设备"
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    mem = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    return True, f"{name} | SM {cap[0]}.{cap[1]} | {mem:.1f} GB"


def c_spec():
    """能不能在 gpulab 的规格表里匹配上（匹配不上就算不出利用率）。"""
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from gpulab import specs
    except Exception as e:  # noqa: BLE001
        return False, f"无法导入 gpulab.specs: {e}"
    try:
        s = specs.detect()
    except Exception as e:  # noqa: BLE001
        return False, str(e).split("\n")[0]
    return True, f"{s.name} | 峰值带宽 {s.bw_gbps:.0f} GB/s | 拐点 {s.knee_intensity('fp16'):.0f} FLOPs/Byte"


def c_triton():
    if importlib.util.find_spec("triton") is None:
        return False, "未安装（W5 写 FlashAttention 要用）"
    import triton
    return True, triton.__version__


def c_ninja():
    """ninja 是 load_inline 的构建后端。

    注意：torch 是用 subprocess 调用 **ninja 可执行文件**，不是 import ninja。
    所以光 pip install 不够，可执行文件必须在 PATH 上 ——
    没激活 conda 环境时就会踩这个坑（症状：pip 装成功了但编译仍报
    "Ninja is required to load C++ extensions"）。
    """
    p = shutil.which("ninja")
    if p:
        out = subprocess.run([p, "--version"], capture_output=True, text=True).stdout.strip()
        return True, f"{p}  v{out}"
    env_bin = Path(sys.executable).parent / "ninja"
    if env_bin.is_file():
        # 已安装，只是不在 PATH 上。harness（bench_kernels.py）会自动把它补进 PATH，
        # 所以这里算通过，但把注意事项写清楚。
        return True, (f"{env_bin}\n"
                      f"        ⚠ 不在 PATH 上 —— 未激活环境时 harness 会自动补；"
                      f"但直接跑 ncu/nsys 包 python 时建议先 conda activate")
    return False, "未找到 ninja 可执行文件"


def _which(cmd, extra_paths=()):
    p = shutil.which(cmd)
    if p:
        return p
    for d in extra_paths:
        cand = os.path.join(d, cmd)
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return None


def c_nvcc():
    p = _which("nvcc", ("/usr/local/cuda/bin",))
    if not p:
        return False, "未找到 nvcc（CUDA Toolkit 未装或 PATH 没生效）"
    out = subprocess.run([p, "--version"], capture_output=True, text=True).stdout
    ver = next((ln.strip() for ln in out.splitlines() if "release" in ln), "")
    return True, f"{p}  {ver}"


def c_ncu():
    # 注意：CUDA Toolkit 自带的 ncu 在 /usr/local/cuda/bin 下，
    # 而那个目录不一定在 PATH 上（非交互 shell 常见）—— 所以要带回退路径。
    # 踩过的坑：漏了这个回退，明明装好了却报"未找到"。
    p = _which("ncu", ("/usr/local/cuda/bin", "/opt/nvidia/nsight-compute"))
    if not p:
        return False, "未找到 ncu"
    out = subprocess.run([p, "--version"], capture_output=True, text=True).stdout
    ver = ""
    for ln in out.splitlines():
        if "Version" in ln:
            ver = ln.strip()
            break
    return True, f"{p}  {ver}"


def c_ncu_perm():
    """ncu 能不能真的抓到 kernel。

    这一步会实际跑一次 profiling。失败通常是 ERR_NVGPUCTRPERM ——
    需要在 **Windows** 的 NVIDIA 控制面板里开启 GPU 性能计数器。
    """
    p = _which("ncu", ("/usr/local/cuda/bin", "/opt/nvidia/nsight-compute"))
    if not p:
        return False, "ncu 未安装，跳过"
    if importlib.util.find_spec("torch") is None:
        return False, "torch 未安装，跳过"
    code = ("import torch;a=torch.randn(512,512,device='cuda',dtype=torch.float16);"
            "b=a@a;torch.cuda.synchronize()")
    r = subprocess.run(
        [p, "--launch-count", "1", "--metrics", "gpu__time_duration.sum",
         sys.executable, "-c", code],
        capture_output=True, text=True, timeout=300,
    )
    blob = (r.stdout or "") + (r.stderr or "")
    if "ERR_NVGPUCTRPERM" in blob:
        return False, ("权限被拒（ERR_NVGPUCTRPERM）—— 需在 Windows 的 NVIDIA 控制面板\n"
                       "        开启：桌面→启用开发者设置→开发者→管理 GPU 性能计数器")
    if r.returncode == 0 and "gpu__time_duration" in blob:
        return True, "可以正常采集 GPU 性能计数器"
    tail = [ln for ln in blob.strip().splitlines() if ln.strip()][-2:]
    return False, " / ".join(tail) if tail else f"返回码 {r.returncode}"


def c_nsys():
    p = _which("nsys")
    return (True, p) if p else (False, "未找到 nsys（Nsight Systems 未装，可选）")


def c_smi():
    p = shutil.which("nvidia-smi")
    if not p:
        return False, "未找到 nvidia-smi（Windows 侧驱动或 WSL GPU 直通有问题）"
    out = subprocess.run(
        [p, "--query-gpu=name,memory.total,power.limit,clocks.max.sm",
         "--format=csv,noheader"],
        capture_output=True, text=True,
    ).stdout.strip()
    return True, out


def c_writable():
    """代码是不是放在 /mnt/ 下 —— 那样编译会非常慢。"""
    here = os.path.abspath(os.path.dirname(__file__))
    if here.startswith("/mnt/"):
        return False, f"项目在 Windows 盘上（{here}），编译/IO 会慢几倍到几十倍"
    return True, here


# ---------------------------------------------------------------------------

def main() -> int:
    checks = [
        ("Python 版本", c_python,
         "装 Python 3.9+"),
        ("项目位置", c_writable,
         "把 aigpu-lab 移到 WSL 内的 ~/projects/ 下，不要在 /mnt/c 或 /mnt/e 里编译"),
        ("PyTorch", c_torch,
         "pip install torch --index-url https://download.pytorch.org/whl/cu128"),
        ("CUDA 可用", c_cuda,
         "确认在 WSL 里（不是 WSL1）、Windows 侧驱动正常、且装的是 CUDA 版 torch 而不是 CPU 版"),
        ("显卡识别", c_device,
         "检查 /dev/dxg 是否存在、WSL 是否为版本 2"),
        ("gpulab 规格表", c_spec,
         "把这张卡的峰值带宽/FP16 吞吐补进 gpulab/specs.py 的 GPU_SPECS"),
        ("Triton", c_triton,
         "pip install triton"),
        ("ninja（编译 kernel 必需）", c_ninja,
         "pip install ninja，然后 conda activate <你的环境> 重开终端\n"
         "        （torch 用 subprocess 调 ninja 可执行文件，必须让它在 PATH 上）"),
        ("CUDA Toolkit (nvcc)", c_nvcc,
         "按《环境搭建清单》第 3 步装 cuda-toolkit-12-8，并把 /usr/local/cuda/bin 加进 PATH"),
        ("Nsight Compute (ncu)", c_ncu,
         "ncu 通常随 CUDA Toolkit 一起装在 /usr/local/cuda/bin。\n"
         "        ⚠️ 不要用 `sudo apt install nsight-compute` —— 那个源里的版本是 2021.3.1，\n"
         "           不支持 Ada（SM 8.9）。装 CUDA Toolkit 时自带的是新版。"),
        ("ncu 采集权限", c_ncu_perm,
         "去 Windows 的 NVIDIA 控制面板 → 桌面（启用开发者设置）→ 开发者 →\n"
         "        管理 GPU 性能计数器 → 允许所有用户，然后 wsl --shutdown 重启"),
        ("Nsight Systems (nsys)", c_nsys,
         "sudo apt install nsight-systems（可选，但做时间线分析很有用）"),
        ("nvidia-smi", c_smi,
         "检查 Windows 侧驱动与 WSL 版本"),
    ]

    print("=" * 74)
    print("环境自检")
    print("=" * 74)
    for name, fn, fix in checks:
        check(name, fn, fix)

    width = max(len(r[0]) for r in RESULTS)
    failed = []
    for name, ok, detail, fix in RESULTS:
        mark = "[OK]  " if ok else "[FAIL]"
        print(f"{mark} {name:<{width}}  {detail}")
        if not ok:
            failed.append((name, fix))

    print("=" * 74)
    if not failed:
        print("全部通过。可以跑：")
        print("    python gpu_watch.py -- python run_baseline.py")
        return 0

    print(f"有 {len(failed)} 项未通过，按顺序修：\n")
    for i, (name, fix) in enumerate(failed, 1):
        print(f"{i}. {name}")
        print(f"   → {fix}\n")
    print("修完再跑一次 python check_env.py 确认。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
