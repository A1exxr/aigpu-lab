"""正确性校验：把你的 kernel 输出和 PyTorch 参考实现对比。

**为什么加速比不能单独报告：**
    一个算错的 kernel 可以无限快。任何性能数字都必须和
    "输出正确" 绑定在一起，否则面试官第一句就是"那结果对吗？"
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def check(ref: torch.Tensor, cand: torch.Tensor,
          atol: float = 1e-2, rtol: float = 1e-2,
          name: str = "", verbose: bool = True) -> dict:
    """对比参考输出 ``ref`` 与候选输出 ``cand``。

    返回 dict，含：
        ok           : 是否通过 allclose
        max_abs_err  : 最大绝对误差
        max_rel_err  : 最大相对误差（分母 clamp 到 1e-6，避免除零）
        cosine_sim   : 余弦相似度（低精度下比逐元素误差更宽容，也更常被引用）
        shape_ok     : 形状是否一致
    """
    result = {
        "name": name or "unnamed",
        "ok": False,
        "shape_ok": tuple(ref.shape) == tuple(cand.shape),
        "max_abs_err": float("nan"),
        "max_rel_err": float("nan"),
        "cosine_sim": float("nan"),
        "ref_dtype": str(ref.dtype),
        "cand_dtype": str(cand.dtype),
    }

    if not result["shape_ok"]:
        if verbose:
            print(f"[FAIL] {result['name']}: 形状不一致 ref={tuple(ref.shape)} "
                  f"cand={tuple(cand.shape)}")
        return result

    ref_f = ref.detach().float()
    cand_f = cand.detach().float()

    diff = (ref_f - cand_f).abs()
    result["max_abs_err"] = diff.max().item()
    result["max_rel_err"] = (diff / ref_f.abs().clamp_min(1e-6)).max().item()

    a, b = ref_f.flatten(), cand_f.flatten()
    if a.numel() > 1:
        result["cosine_sim"] = F.cosine_similarity(a, b, dim=0).item()

    result["ok"] = torch.allclose(ref_f, cand_f, atol=atol, rtol=rtol)

    if verbose:
        tag = "PASS" if result["ok"] else "FAIL"
        print(
            f"[{tag}] {result['name']:<28} "
            f"max_abs={result['max_abs_err']:.3e}  "
            f"max_rel={result['max_rel_err']:.3e}  "
            f"cos={result['cosine_sim']:.6f}"
        )
    return result


def check_fn(ref_fn, cand_fn, atol: float = 1e-2, rtol: float = 1e-2,
             name: str = "", verbose: bool = True) -> dict:
    """直接对比两个无参函数（内部自己准备输入，保证公平）。"""
    return check(ref_fn(), cand_fn(), atol=atol, rtol=rtol,
                 name=name, verbose=verbose)


def assert_correct(result: dict) -> None:
    """在 benchmark 之前调用：不正确就直接停，别浪费机时跑性能。"""
    if not result["ok"]:
        raise AssertionError(
            f"{result['name']} 输出不正确（max_abs_err={result['max_abs_err']:.3e}）。"
            f"先修正确性，再谈性能。"
        )
