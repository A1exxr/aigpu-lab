"""w11 FlashAttention 反向 · 一键跑全部台阶（0 → 4）

用法（在 kernels/ 目录下）：
    cd ~/projects/aigpu-lab/kernels
    ~/miniconda3/envs/triton/bin/python w11_all.py          # 只跑正确性
    ~/miniconda3/envs/triton/bin/python w11_all.py --bench   # 正确性 + 性能表

文件分工：
    w11_oracle.py        台阶 0：PyTorch 尺子（fwd / bwd_ref / make_inputs）
    w11_bwd_1_block.py   台阶 1：Triton 单块（S=64）
    w11_bwd_2_chunked.py 台阶 2：分块 + atomic（S=1024）
    w11_bwd_3_split.py   台阶 3：拆两 kernel，无 atomic（可复现）
    w11_bwd_4_fp16.py    台阶 4：fp16 + Tensor Core
"""

import sys

from w11_bwd_1_block import test_step1
from w11_bwd_2_chunked import test_step2
from w11_bwd_3_split import test_step3
from w11_bwd_4_fp16 import test_step4
from w11_oracle import test_step0


def main():
    results = {
        "台阶0 PyTorch 尺子": test_step0(),
        "台阶1 单块": test_step1(),
        "台阶2 分块+atomic": test_step2(),
        "台阶3 无 atomic": test_step3(),
        "台阶4 fp16+TC": test_step4(),
    }

    print()
    print("=" * 50)
    print("汇总")
    print("-" * 50)
    for k, v in results.items():
        print(f"  {k:<20} {'✅' if v else '❌'}")
    allok = all(results.values())
    print("=" * 50)
    print("全部通过 ✅" if allok else "有失败项 ❌")

    if "--bench" in sys.argv:
        from w11_bench import main as bench_main

        bench_main()

    return allok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
