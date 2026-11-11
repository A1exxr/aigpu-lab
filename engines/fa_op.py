"""R1：把自研 flash attention 注册成 PyTorch 算子。

现状：`kernels/w5_flash_attn.py` 里 `flash_attn(q, k, v)` 只是一个 Python 函数，
      PyTorch 的 dispatcher 不认识它 —— 不能进 torch.compile 的图，profiler 里没有名字，
      也没法挂 autograd。

目标（四条判据，跑 `python engines/fa_op.py` 全绿才算过）：
  ① torch.ops.aigpu.flash_attn(q, k, v, scale) 与 SDPA 数值一致（fp16，max diff < 2e-3）
  ② torch.compile(fullgraph=True) 能捕获，不 graph break
  ③ torch.profiler 里能看到 `aigpu::flash_attn` 这个名字
  ④ 反向能跑：dq/dk/dv 与 PyTorch 自动求导的 rel err < 2e-2

三条 TODO（每处都有详细提示，按顺序填）。
"""
import sys
from pathlib import Path

import torch

# 让 engines/ 能 import kernels/ 里的东西
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "kernels"))
from w5_flash_attn import flash_attn as triton_fa  # noqa: E402

D_HEAD = 64          # w5 的 kernel 把 D 当 constexpr，只支持这个值
TOL_FWD = 2e-3
TOL_BWD = 2e-2


# ============================================================ TODO 1
# 定义算子：名字必须带命名空间，格式 "命名空间::函数名"
# 提示：
#   - 装饰器用 torch.library.custom_op("aigpu::flash_attn", mutates_args=())
#   - 类型标注是必须的：custom_op 靠它推断 schema
#   - kernel 里 scale 写死成 1/sqrt(D)，所以要先把 q 乘上 (scale * sqrt(D)) 再喂进去
#   - 输入不一定连续，进来先 .contiguous()
#   - 必须检查 dtype/device/D，不合法就 raise（别静默算错）
@torch.library.custom_op("aigpu::flash_attn", mutates_args=())
def flash_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
    raise NotImplementedError("TODO 1: 调 triton_fa，把 scale 折进 q")


# ============================================================ TODO 2
# fake（meta）实现：只算形状，不分配显存。
# 作用：让 torch.compile 能在「没有真实数据」的图阶段推断输出形状。
# 提示：输出和 q 同形状同 dtype 同 device —— 用 q.new_empty(q.shape) 最省事。
@flash_attn.register_fake
def _(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
    raise NotImplementedError("TODO 2: return q.new_empty(q.shape)")


# ============================================================ TODO 3
# 反向：用 torch.library.register_autograd 挂到刚注册的算子上。
# 提示：
#   - 签名 register_autograd("aigpu::flash_attn", backward, setup_context=None)（没有需要保存的上下文时可省）
#   - backward(ctx, grad_out) 里能拿到 grad_out，但拿不到前向的 q/k/v ——
#     所以要用 setup_context 把输入存进 ctx（或用函数式：让 backward 重新接收输入）
#   - 数学（你在 w11 已经推过一遍，这里直接用）：
#         delta = (do * o).sum(-1)                      # 每行一个标量
#         dp    = do @ v.transpose(-1, -2)              # [B,H,S,S]
#         ds    = p * (dp - delta[..., None])           # 逐元素，不是矩阵乘
#         dv    = p.transpose(-1, -2) @ do
#         dq    = (ds @ k) * scale
#         dk    = (ds.transpose(-1, -2) @ q) * scale
#     p 需要重新算：p = softmax(q @ kᵀ * scale)
#   - 先写成纯 torch 版本把链路跑通（能过判据④即可），跑通后再换成 w11 的 Triton 反向
def _backward(ctx, grad_out):
    raise NotImplementedError("TODO 3: 按上面的公式算 dq/dk/dv")


# ============================================================ 判据
def _ref_sdpa(q, k, v, scale):
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=scale)


def _check_fwd(q, k, v, scale):
    out = torch.ops.aigpu.flash_attn(q, k, v, scale)
    ref = _ref_sdpa(q, k, v, scale)
    diff = (out.float() - ref.float()).abs().max().item()
    ok = diff < TOL_FWD
    print(f"① 数值      max|out - SDPA| = {diff:.3e}   (判据 < {TOL_FWD:g})   {'OK' if ok else 'FAIL'}")
    return ok


def _check_compile(q, k, v, scale):
    def f(q, k, v):
        return torch.ops.aigpu.flash_attn(q, k, v, scale)

    try:
        cf = torch.compile(f, fullgraph=True, dynamic=False)
        out = cf(q, k, v)
        ref = _ref_sdpa(q, k, v, scale)
        diff = (out.float() - ref.float()).abs().max().item()
        ok = diff < TOL_FWD
        print(f"② compile   fullgraph=True 通过，max diff = {diff:.3e}          {'OK' if ok else 'FAIL'}")
        return ok
    except Exception as e:                                    # noqa: BLE001
        print(f"② compile   FAIL: {type(e).__name__}: {str(e)[:120]}")
        return False


def _check_profiler(q, k, v, scale):
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        torch.ops.aigpu.flash_attn(q, k, v, scale)
        torch.cuda.synchronize()
    names = {e.key for e in prof.key_averages()}
    hit = any("aigpu::flash_attn" in n for n in names)
    print(f"③ profiler  {'看到 aigpu::flash_attn' if hit else '没看到（名字没进 profiler）'}          {'OK' if hit else 'FAIL'}")
    return hit


def _check_bwd(q, k, v, scale):
    q1, k1, v1 = (x.clone().requires_grad_(True) for x in (q, k, v))
    ref = _ref_sdpa(q1, k1, v1, scale)
    g = torch.randn_like(ref)
    ref.backward(g)
    ref_grads = [q1.grad.clone(), k1.grad.clone(), v1.grad.clone()]

    q2, k2, v2 = (x.clone().requires_grad_(True) for x in (q, k, v))
    out = torch.ops.aigpu.flash_attn(q2, k2, v2, scale)
    out.backward(g)
    our_grads = [q2.grad, k2.grad, v2.grad]

    errs = []
    for name, a, b in zip("qkv", our_grads, ref_grads):
        if a is None:
            print(f"④ 反向      d{name} 是 None（没接上 autograd）          FAIL")
            return False
        denom = b.abs().mean().clamp_min(1e-6)
        errs.append(((a - b).abs().mean() / denom).item())
    ok = all(e < TOL_BWD for e in errs)
    print(f"④ 反向      rel err dq/dk/dv = {errs[0]:.2e}/{errs[1]:.2e}/{errs[2]:.2e}   "
          f"(判据 < {TOL_BWD:g})   {'OK' if ok else 'FAIL'}")
    return ok


def main():
    torch.manual_seed(0)
    B, H, S, D = 2, 4, 256, D_HEAD
    q, k, v = (torch.randn(B, H, S, D, device="cuda", dtype=torch.float16) for _ in range(3))
    scale = 1.0 / D ** 0.5

    print(f"shape q/k/v = {tuple(q.shape)} fp16, scale = {scale}")
    results = [
        _check_fwd(q, k, v, scale),
        _check_compile(q, k, v, scale),
        _check_profiler(q, k, v, scale),
        _check_bwd(q, k, v, scale),
    ]
    print()
    print("PASS" if all(results) else "FAIL", f"({sum(results)}/4)")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
