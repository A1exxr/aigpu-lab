#!/usr/bin/env python3
"""从已编译的 .so / .cubin 里 dump SASS，并按函数统计"关键指令"条数。

用法：
    python tools/_sass.py                 # 自动找 ~/.cache/torch_extensions 下最新的 .so
    python tools/_sass.py path/to/x.so    # 指定文件
    python tools/_sass.py path/to/x.cubin # 也支持 cubin

需要 cuobjdump：export PATH=/usr/local/cuda/bin:$PATH
判读说明见 docs/01-概念/08-SASS怎么读.md
"""
import os
import re
import shutil
import subprocess
import sys
from glob import glob

# 说明：SASS 里 32 位访问写成裸的 LDG.E / STG.E（没有 .32 后缀），
#       128/64 位显式写 .128 / .64，窄于 32 位的写 .U16 / .U8。
TERMS = [
    ("LDG总", r"\bLDG\.E"),
    ("LDG128", r"\bLDG\.E\.128"),
    ("LDG64", r"\bLDG\.E\.64"),
    ("LDGU16", r"\bLDG\.E\.U16"),
    ("STG总", r"\bSTG\.E"),
    ("STGU16", r"\bSTG\.E\.U16"),
    ("LDL", r"\bLDL\b"),
    ("STL", r"\bSTL\b"),
    ("LDS", r"\bLDS"),
    ("STS", r"\bSTS"),
    ("HMMA", r"\bHMMA"),
    ("LDSM", r"\bLDSM"),
    ("FFMA", r"\bFFMA"),
    ("BAR", r"BAR\.SYNC"),
    ("LDGSTS", r"\bLDGSTS"),
    ("SHFL", r"\bSHFL"),
    ("判定P", r"^\s*@!?P\d"),
]


def find_so():
    base = os.path.expanduser("~/.cache/torch_extensions")
    cands = glob(os.path.join(base, "*", "*", "*.so"))
    if not cands:
        sys.exit(f"没找到 .so，试着显式传路径：python tools/_sass.py xxx.so\n（找过 {base}）")
    return max(cands, key=os.path.getmtime)


def dump_sass(path):
    if not shutil.which("cuobjdump"):
        sys.exit("cuobjdump 不在 PATH：export PATH=/usr/local/cuda/bin:$PATH")
    return subprocess.run(
        ["cuobjdump", "-sass", path], capture_output=True, text=True, check=True
    ).stdout


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else find_so()
    print(f"# 文件: {path}\n")
    text = dump_sass(path)

    blocks = re.split(r"^\s*Function : ", text, flags=re.M)
    if len(blocks) <= 1:
        print("没解析到 Function 块，前 40 行：")
        print("\n".join(text.splitlines()[:40]))
        return

    header = "kernel".ljust(34) + "".join(t.rjust(9) for t, _ in TERMS)
    print(header)
    print("-" * len(header))
    for blk in blocks[1:]:
        name = blk.splitlines()[0].strip().split("(")[0]
        counts = [len(re.findall(rx, blk, flags=re.M)) for _, rx in TERMS]
        print(name[-32:].ljust(34) + "".join(str(c).rjust(9) for c in counts))

    print("\n# 判读：")
    print("#   LDG128 > 0 且 LDGU16/32位 == 0   → 向量化生效（每线程 16 B）")
    print("#   LDGU16 多                          → 一元素一条指令，2 B/次（扇区浪费）")
    print("#   LDL/STL > 0                        → 有 local memory（取局部变量地址或溢出）")
    print("#   HMMA > 0                           → 真在用张量核心")
    print("#   LDS/STS/BAR == 0                   → 完全没用共享内存")
    print("#   LDGSTS > 0                         → cp.async 生效")


if __name__ == "__main__":
    main()
