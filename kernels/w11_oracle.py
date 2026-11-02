"""w11 FlashAttention 反向 · 台阶 0：PyTorch 手写参考实现（oracle 尺子）

这个文件只做一件事：**给出"正确答案"**，供后面的 Triton 版本对照。
它自己可以被单独验证（跑一次跟 autograd 比，差 1e-16）。

"""

import math
import torch

# ==========================================================================
#                          两个纯函数（被别的文件 import）
# ==========================================================================
def fwd(q, k, v):
    B, H, S, D = q.shape
    scale = 1.0 / math.sqrt(D)
    s = (q @ k.transpose(-1, -2)) * scale     # (B,H,S,S)
    lse = torch.logsumexp(s, dim=-1)          # (B,H,S)
    p = torch.exp(s - lse[..., None])         # (B,H,S,S)
    o = p @ v                                 # (B,H,S,D)
    return s, p, o, lse


def bwd_ref(q, k, v, do, scale=None):
    D = q.shape[-1]
    s = (q @ k.transpose(-1, -2)) / math.sqrt(D)                    # (B, H, S, S)

    # 每行 log-sum-exp；沿着最后一维S做归约，再保留该维为1
    lse = torch.logsumexp(s, dim=-1, keepdim=True)                  # (B,H,S,S) → (B,H,S,1)
    p = torch.exp(s - lse)                                          # (B,H,S,S) − (B,H,S,1) → (B,H,S,S)
    o = p @ v                                                       # 只为让下面 delta 那行成立

    dv = p.transpose(-1, -2) @ do                       # (B,H,S,S)ᵀ @ (B,H,S,D) → (B,H,S,D)

    dp = do @ v.transpose(-1, -2)                       # (B,H,S,D) @ (B,H,D,S) → (B,H,S,S)

    # 不能用 @，@ 是把左边最后一维和右边倒数第二维乘完再加总（会消掉一维）；
    # * 是逐元素配对，形状不变
    delta = (do * o).sum(dim=-1, keepdim=True)       # 逐元素乘，(B,H,S,D)，再沿着D维归约，并保留为1 -> (B,H,S,1)

    # ds 要的形状是 (B,H,S,S)，和 p 一样
    ds = p * (dp - delta)                           # (B,H,S,S)

    dq = ds @ k / math.sqrt(D)                      # (B,H,S,S) @ (B,H,S,D) → (B,H,S,D)

    dk = ds.transpose(-1, -2) @ q / math.sqrt(D)    # (B,H,S,S)ᵀ @ (B,H,S,D) → (B,H,S,D)
    return dq, dk, dv, p, lse, ds


def make_inputs(B, H, S, D, seed=0, dtype=torch.float32, requires_grad=False):
    """固定种子造 q/k/v/do，方便不同版本用同一组数据。"""
    g = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
    g.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    shape = (B, H, S, D)
    q = torch.randn(shape, generator=g, device=dev, dtype=dtype)
    k = torch.randn(shape, generator=g, device=dev, dtype=dtype)
    v = torch.randn(shape, generator=g, device=dev, dtype=dtype)
    do = torch.randn(shape, generator=g, device=dev, dtype=dtype)
    if requires_grad:
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)
    return q, k, v, do


# ==========================================================================
#                    台阶 0 自检：oracle 自己跟 autograd 比
# ==========================================================================
def test_step0():
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)
    B, H, S, D = 1, 8, 128, 64
    q, k, v, do = make_inputs(B, H, S, D, seed=0, dtype=torch.float64, requires_grad=True)

    _, p_ref, o, _ = fwd(q, k, v)
    dq_ref, dk_ref, dv_ref = torch.autograd.grad(o, (q, k, v), do)
    dq, dk, dv, p, lse, ds = bwd_ref(q, k, v, do)

    print(f"台阶 0 · device={q.device}  B={B} H={H} S={S} D={D}  dtype=torch.float64")

    # 自检 0：用 lse 重算的 p 与 softmax 一致
    p_err = (p - p_ref).abs().max().item()
    print(f"[自检0] max|p - softmax(s)|       = {p_err:.3e}   {'OK' if p_err < 1e-12 else 'FAIL'}")

    # 自检 1：ds 行和为 0（Jacobian 项没漏）
    row_sum = ds.sum(dim=-1).abs().max().item()
    print(f"[自检1] max|ds.sum(-1)|           = {row_sum:.3e}   {'OK' if row_sum < 1e-9 else 'FAIL'}")

    # 自检 2：delta 两种算法等价
    delta_alt = (do * o).sum(dim=-1)
    delta_mine = (p * (do @ v.transpose(-1, -2))).sum(dim=-1)
    d_err = (delta_mine - delta_alt).abs().max().item()
    print(f"[自检2] max|delta - rowsum(do∘o)| = {d_err:.3e}   {'OK' if d_err < 1e-10 else 'FAIL'}")

    print()
    ok = True
    for name, mine, ref in (("dq", dq, dq_ref), ("dk", dk, dk_ref), ("dv", dv, dv_ref)):
        err = (mine - ref).abs().max().item()
        rel = err / ref.abs().max().item()
        good = err < 1e-5
        ok &= good
        print(f"{name}: max_abs = {err:.3e}   rel = {rel:.3e}   {'OK' if good else 'FAIL'}")

    print("\n" + ("台阶 0 通过：oracle 可信，可以当尺子用 ✅" if ok else "台阶 0 未通过 —— 别拿它当标准答案"))
    return ok


if __name__ == "__main__":
    test_step0()
