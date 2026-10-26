"""只编译、不跑 bench 的小工具（写 kernel 时的快速反馈回路）。

用法（必须在仓库根目录跑，或用下面的绝对路径）：
    ~/miniconda3/envs/triton/bin/python kernels/_compile_only.py paged_attn
    ~/miniconda3/envs/triton/bin/python kernels/_compile_only.py paged_attn --v   # 出错时打印完整 nvcc 日志

退出码：0 = 编译通过；1 = 编译失败。
"""
import os
import sys

ROOT = os.path.expanduser("~/projects/aigpu-lab")
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "kernels"))
# 让 ninja 等工具可见：把当前 python 所在环境的 bin 放进 PATH
os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")

import bench_kernels as bk  # noqa: E402

name = "paged_attn"
verbose = "--v" in sys.argv
for a in sys.argv[1:]:
    if not a.startswith("--"):
        name = a

spec = next((s for s in bk.REGISTRY if s.name == name), None)
if spec is None:
    print(f"没有这个 kernel：{name}")
    print("可选：", ", ".join(s.name for s in bk.REGISTRY))
    sys.exit(2)
print(f"编译 {name}（{spec.cuda_file}）...")
mod, err = bk.build_module(spec, verbose)
if mod is not None:
    print("✅ 编译通过")
    sys.exit(0)

print(f"❌ {err}")
if not verbose:
    print("提示：加 --v 重跑，能看到 nvcc 的具体报错行。")
sys.exit(1)
