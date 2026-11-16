"""
w12c_qk.py —— w12c_mma.cu（Stage A：S = Q·Kᵀ）的 driver

跑法：
    cd ~/projects/aigpu-lab
    ~/miniconda3/envs/triton/bin/python kernels/w12c_qk.py

看什么：
    1) 是否 PASS（max|diff| <= 2e-3；fp16 输入、fp32 累加、K 维度 64）
    2) 只有 FAIL 时才跑 [2] 布局诊断：把 S 和几种"错位版本"比，指向是哪一类布局错

2026-11-15 首跑结果：max|diff| = 5.722e-06  PASS（16 条 HMMA + 16 条 LDSM.x2 + 4 条 LDSM.x4）
"""

import sys
from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline

HERE = Path(__file__).resolve().parent
CU = (HERE / "w12c_mma.cu").read_text(encoding="utf-8")

CPP = "torch::Tensor qk_mma(torch::Tensor q, torch::Tensor k);"

print("[build] compiling w12c_mma ... (第一次 40-70 s)", flush=True)
mod = load_inline(
    name="w12c_mma",
    cpp_sources=CPP,
    cuda_sources=CU,
    functions=["qk_mma"],
    extra_cuda_cflags=["-O3", "-lineinfo"],
    verbose=False,
)
print("[build] ok", flush=True)


def main():
    torch.manual_seed(0)
    B, H, Hkv, S, D = 2, 8, 2, 512, 64
    q = (torch.randn(B, H, S, D, device="cuda") * 0.5).half()
    k = (torch.randn(B, Hkv, S, D, device="cuda") * 0.5).half()

    Sgpu = mod.qk_mma(q, k)

    grp = H // Hkv
    kk = k.float().repeat_interleave(grp, dim=1)          # GQA 展开
    ref = q.float() @ kk.transpose(-1, -2)                # [B,H,S,S]

    diff = (Sgpu - ref).abs()
    mx = diff.max().item()
    print("\n[1] 正确性（Stage A：只比 S，不缩放、不 mask）")
    print(f"  S shape = {tuple(Sgpu.shape)}   max|diff| = {mx:.3e}")
    print(f"  S[0,0,0,:4]   = {[round(v,4) for v in Sgpu[0,0,0,:4].tolist()]}")
    print(f"  ref[0,0,0,:4] = {[round(v,4) for v in ref[0,0,0,:4].tolist()]}")
    print(f"  S[0,0,64,:4]  = {[round(v,4) for v in Sgpu[0,0,64,:4].tolist()]}")
    print(f"  ref[0,0,64,:4]= {[round(v,4) for v in ref[0,0,64,:4].tolist()]}")

    ok = mx <= 2e-3

    # 只有失败时才做诊断（PASS 时这些检查没意义，且容易写错）
    if not ok:
        print("\n[2] 布局诊断（哪一行最小 → 就是哪一类错位）")
        dev = Sgpu.device
        pair = torch.arange(S, device=dev) ^ 1                       # 0<->1, 2<->3 ...
        cand = {
            "行上下半互换（c2,c3 写到别的行）": (Sgpu - ref.flip(-2)).abs().max().item(),
            "列两两互换（t 步长错）": (Sgpu - ref[..., pair]).abs().max().item(),
            "列每 8 个一组错位": (Sgpu - ref.roll(8, dims=-1)).abs().max().item(),
            "S 被当成 Sᵀ": (Sgpu - ref.transpose(-1, -2)).abs().max().item(),
            "全 0（mma 没跑 / 没写回）": Sgpu.abs().max().item(),
        }
        for name, v in cand.items():
            print(f"  {name:34s} {v:.3e}")

    print("\n[3] 结论")
    print("  " + ("PASS —— QK 的 mma 通了，进 Stage B（mask + softmax + PV）" if ok
                  else "FAIL —— 先看 [2] 哪一行最小"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
