"""W2/W3 交付物生成器：编译 → 验证 → 测量你自己写的 CUDA kernel。

分工原则
--------
  kernel 代码    → 你写（这是你要练的东西）
  编译/验证/测量/出报告 → 我写（这是样板工程，不体现你的能力）

用法
----
    # 先自检 harness 本身能不能跑（不需要你写任何 CUDA）
    python kernels/bench_kernels.py --self-test

    # 编译并测量所有已存在的 kernel
    python kernels/bench_kernels.py

    # 只测某一个
    python kernels/bench_kernels.py --only rmsnorm

产出
----
    results/kernels.md            对比表（可直接贴进 README）
    results/fig_roofline_kernels.png

工作流
------
下面 REGISTRY 里的条目**从第一天就全部存在**。你每写完一个 .cu 文件，
对应条目就会从「文件不存在」变成真实的测量结果。
每周跑一次，看着它一个个变绿。
"""
from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

# 允许从 aigpu-lab 根目录直接运行本脚本
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gpulab import plotting, report, specs, timer, verify, workloads  # noqa: E402

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
DEV = "cuda"


# ---------------------------------------------------------------------------
# 参考实现（PyTorch）—— 你的 kernel 要和它们对得上
# ---------------------------------------------------------------------------

def ref_rmsnorm(x, w, eps=1e-6):
    var = x.float().pow(2).mean(dim=-1, keepdim=True)
    return (x.float() * torch.rsqrt(var + eps)).to(x.dtype) * w


def ref_softmax(x):
    return F.softmax(x, dim=-1)


def ref_gelu(x):
    return F.gelu(x)


def ref_fused_rmsnorm_residual(x, residual, w, eps=1e-6):
    """融合算子的参考实现：先加残差再归一化（两步，会多读写一次 HBM）。"""
    h = x + residual
    return ref_rmsnorm(h, w, eps)

# w2_reduce.cu
def ref_reduce(x):
    return x.sum()


def ref_attn_naive(Q, K, V):
    """w4_attn_naive.cu 的参考实现。

    用 PyTorch 的 SDPA，但**显式按 naive 的数学写一遍**更稳妥（SDPA 可能走
    flash/mem-efficient 后端，数值上等价但实现不同）。这里用显式公式：

        S = Q @ K.transpose(-1, -2) / sqrt(D)
        P = softmax(S, dim=-1)
        O = P @ V

    全程在 fp32 里算，最后转回输入 dtype，避免 fp16 中间结果把 1e-2 的
    容差吃满。
    """
    d = Q.size(-1)
    scale = 1.0 / math.sqrt(d)
    q = Q.float()
    k = K.float()
    v = V.float()
    s = torch.matmul(q, k.transpose(-1, -2)) * scale
    p = torch.softmax(s, dim=-1)
    return torch.matmul(p, v).to(Q.dtype)




# ---------------------------------------------------------------------------
# 规格定义
# ---------------------------------------------------------------------------

@dataclass
class Case:
    """一个测试配置。"""
    label: str
    shape: tuple
    dtype: str
    make_inputs: object      # (shape, torch.dtype, device) -> dict
    cuda_call: object        # (module, inputs) -> Tensor
    ref_call: object         # (inputs) -> Tensor
    nbytes: object           # (shape, torch.dtype) -> int
    flops: object = None     # (shape, torch.dtype) -> float


@dataclass
class KernelSpec:
    name: str
    cuda_file: str
    cpp_signature: str
    functions: list
    cases: list
    note: str = ""
    cuda_source: str = ""     # 若设置，直接用这段源码，忽略 cuda_file（内置自检用）


# ---------------------------------------------------------------------------
# ★ 在这里注册你的 kernel ★
#    （条目现在就可以全部留着，没写的文件会被跳过并提示）
# ---------------------------------------------------------------------------

RMSNORM_HP = (1, 2048, 4096)

REDUCE_HP = (4194304,)  # w2_reduce.cu

def _rmsnorm_inputs(shape, dtype, device):
    x = torch.randn(*shape, device=device, dtype=dtype)
    w = torch.randn(shape[-1], device=device, dtype=dtype)
    return {"x": x, "w": w}


def ref_gemm(A, B):
    """w6_gemm.cu 的参考实现：C = A @ B（fp16 输入，fp32 累加后转回 fp16）。"""
    return torch.matmul(A.float(), B.float()).to(A.dtype)


def ref_paged_attn(q, k_cache, v_cache, block_table, context_lens):
    """PagedAttention decode 的参考实现（fp32 计算；先把逻辑块 gather 成连续 K/V 再算）。"""
    B, H, D = q.shape
    bs = k_cache.size(1)
    out = torch.zeros(B, H, D, dtype=torch.float32, device=q.device)
    for b in range(B):
        L = int(context_lens[b].item())
        nb = (L + bs - 1) // bs
        phys = block_table[b, :nb]                      # [nb] 物理块号（乱序）
        k = k_cache[phys].reshape(-1, H, D)[:L].float()
        v = v_cache[phys].reshape(-1, H, D)[:L].float()
        scores = torch.einsum("hd,lhd->hl", q[b].float(), k)
        probs = torch.softmax(scores, dim=-1)
        out[b] = torch.einsum("hl,lhd->hd", probs, v)
    return out.to(q.dtype)


def _make_paged_inputs(shape, dtype, dev):
    """shape = (B, H, ctx, D)；块表用随机置换，杜绝"物理块恰好连续"的巧合。"""
    B, H, ctx, D = shape
    bs = 16
    maxb = (ctx + bs - 1) // bs
    nb = B * maxb
    q = torch.randn(B, H, D, device=dev, dtype=dtype)
    k_cache = torch.randn(nb, bs, H, D, device=dev, dtype=dtype)
    v_cache = torch.randn(nb, bs, H, D, device=dev, dtype=dtype)
    perm = torch.randperm(nb, device=dev)
    block_table = perm[: B * maxb].reshape(B, maxb).to(torch.int32).contiguous()
    context_lens = torch.full((B,), ctx, dtype=torch.int32, device=dev)
    return {"q": q, "k_cache": k_cache, "v_cache": v_cache,
            "block_table": block_table, "context_lens": context_lens}


def _make_paged_int8_inputs(shape, dtype, dev):
    """K cache 为 int8 + per-(token,head) 对称量化 scale；V 仍是 fp16。"""
    B, H, ctx, D = shape
    bs = 16
    maxb = (ctx + bs - 1) // bs
    nb = B * maxb
    q = torch.randn(B, H, D, device=dev, dtype=dtype)
    k = torch.randn(nb, bs, H, D, device=dev, dtype=torch.float32)
    absmax = k.abs().amax(dim=-1, keepdim=True).clamp_min(1e-5)
    scale = (absmax / 127.0).to(torch.float16)                 # [nb,bs,H,1]
    k_q = torch.round(k / scale.float()).clamp(-127, 127).to(torch.int8)
    v_cache = torch.randn(nb, bs, H, D, device=dev, dtype=dtype)
    perm = torch.randperm(nb, device=dev)
    block_table = perm[: B * maxb].reshape(B, maxb).to(torch.int32).contiguous()
    context_lens = torch.full((B,), ctx, dtype=torch.int32, device=dev)
    return {"q": q, "k_q": k_q, "k_scale": scale.squeeze(-1).contiguous(),
            "v_cache": v_cache, "block_table": block_table,
            "context_lens": context_lens}


def _make_paged_int8kv_inputs(shape, dtype, dev):
    """K 和 V 都是 int8 + per-(token,head) 对称 scale；q/out 仍是 fp16。"""
    B, H, ctx, D = shape
    bs = 16
    maxb = (ctx + bs - 1) // bs
    nb = B * maxb
    q = torch.randn(B, H, D, device=dev, dtype=dtype)

    def _quant(x):
        absmax = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-5)
        scale = (absmax / 127.0).to(torch.float16)
        q8 = torch.round(x / scale.float()).clamp(-127, 127).to(torch.int8)
        return q8, scale.squeeze(-1).contiguous()

    k = torch.randn(nb, bs, H, D, device=dev, dtype=torch.float32)
    v = torch.randn(nb, bs, H, D, device=dev, dtype=torch.float32)
    k_q, k_scale = _quant(k)
    v_q, v_scale = _quant(v)

    perm = torch.randperm(nb, device=dev)
    block_table = perm[: B * maxb].reshape(B, maxb).to(torch.int32).contiguous()
    context_lens = torch.full((B,), ctx, dtype=torch.int32, device=dev)
    return {"q": q, "k_q": k_q, "k_scale": k_scale, "v_q": v_q, "v_scale": v_scale,
            "block_table": block_table, "context_lens": context_lens}


def ref_paged_attn_int8kv(q, k_q, k_scale, v_q, v_scale, block_table, context_lens):
    """K/V 都先反量化回 fp32，再走 fp16 参考实现。"""
    k_cache = k_q.float() * k_scale.float().unsqueeze(-1)
    v_cache = v_q.float() * v_scale.float().unsqueeze(-1)
    return ref_paged_attn(q, k_cache, v_cache, block_table, context_lens)


def ref_paged_attn_int8(q, k_q, k_scale, v_cache, block_table, context_lens):
    """先把 int8 K 反量化回 fp32，再走 fp16 的参考实现。"""
    k_cache = k_q.float() * k_scale.float().unsqueeze(-1)
    return ref_paged_attn(q, k_cache, v_cache, block_table, context_lens)


REGISTRY: list[KernelSpec] = [
    KernelSpec(
        name="rmsnorm",
        cuda_file="kernels/w2_rmsnorm.cu",
        cpp_signature=(
            "torch::Tensor rmsnorm(torch::Tensor x, torch::Tensor w, double eps);"
        ),
        functions=["rmsnorm"],
        note="W2：纯 memory-bound，用来练带宽利用率（目标 > 70%）",
        cases=[
            Case(
                label="(1,2048,4096) fp16",
                shape=RMSNORM_HP, dtype="fp16",
                make_inputs=_rmsnorm_inputs,
                cuda_call=lambda mod, inp: mod.rmsnorm(inp["x"], inp["w"], 1e-6),
                ref_call=lambda inp: ref_rmsnorm(inp["x"], inp["w"]),
                nbytes=lambda s, d: workloads.bytes_layer_norm(s, d, affine=False)
                                    + s[-1] * workloads.elem_bytes(d),
                flops=lambda s, d: 6.0 * workloads.numel(s),
            ),
        ],
    ),
    KernelSpec(
        name="softmax",
        cuda_file="kernels/w2_softmax.cu",
        cpp_signature="torch::Tensor softmax_lastdim(torch::Tensor x);",
        functions=["softmax_lastdim"],
        note="W2：规约练习（max → exp → sum → div）",
        cases=[
            Case(
                label="(1,2048,4096) fp16",
                shape=RMSNORM_HP, dtype="fp16",
                make_inputs=lambda s, d, dev: {"x": torch.randn(*s, device=dev, dtype=d)},
                cuda_call=lambda mod, inp: mod.softmax_lastdim(inp["x"]),
                ref_call=lambda inp: ref_softmax(inp["x"]),
                nbytes=lambda s, d: workloads.bytes_softmax(s, d),
                flops=lambda s, d: 5.0 * workloads.numel(s),
            ),
        ],
    ),
    KernelSpec(
        name="gelu",
        cuda_file="kernels/w2_gelu.cu",
        cpp_signature="torch::Tensor gelu(torch::Tensor x);",
        functions=["gelu"],
        note="W2：element-wise，练 float4 向量化访存",
        cases=[
            Case(
                label="(1,2048,4096) fp16",
                shape=RMSNORM_HP, dtype="fp16",
                make_inputs=lambda s, d, dev: {"x": torch.randn(*s, device=dev, dtype=d)},
                cuda_call=lambda mod, inp: mod.gelu(inp["x"]),
                ref_call=lambda inp: ref_gelu(inp["x"]),
                nbytes=lambda s, d: workloads.bytes_elementwise(s, d, 1, 1),
                flops=lambda s, d: 8.0 * workloads.numel(s),
            ),
        ],
    ),
    KernelSpec(
        name="fused_rmsnorm_residual",
        cuda_file="kernels/w3_fused_rmsnorm_residual.cu",
        cpp_signature=(
            "torch::Tensor fused_rmsnorm_residual("
            "torch::Tensor x, torch::Tensor residual, torch::Tensor w, double eps);"
        ),
        functions=["fused_rmsnorm_residual"],
        note="W3：融合算子，核心叙事是「减少 HBM 往返」。要对比分步实现",
        cases=[
            Case(
                label="(1,2048,4096) fp16",
                shape=RMSNORM_HP, dtype="fp16",
                make_inputs=lambda s, d, dev: {
                    "x": torch.randn(*s, device=dev, dtype=d),
                    "r": torch.randn(*s, device=dev, dtype=d),
                    "w": torch.randn(s[-1], device=dev, dtype=d),
                },
                cuda_call=lambda mod, inp: mod.fused_rmsnorm_residual(
                    inp["x"], inp["r"], inp["w"], 1e-6),
                ref_call=lambda inp: ref_fused_rmsnorm_residual(
                    inp["x"], inp["r"], inp["w"]),
                # 融合版：读 x + 读 r + 写 out = 3N；分步版要多一轮 = 5N
                nbytes=lambda s, d: 3 * workloads.numel(s) * workloads.elem_bytes(d),
                flops=lambda s, d: 7.0 * workloads.numel(s),
            ),
        ],
    ),
    KernelSpec(
        name="two_step_rmsnorm_residual",
        cuda_file="kernels/w3_fused_rmsnorm_residual.cu",
        cpp_signature=(
            "torch::Tensor two_step_rmsnorm_residual("
            "torch::Tensor x, torch::Tensor residual, torch::Tensor w, double eps);"
        ),
        functions=["two_step_rmsnorm_residual"],
        note="W3：分步对照版（add + rmsnorm，两个 kernel），h 读两遍 = 6N",
        cases=[
            Case(
                label="(1,2048,4096) fp16",
                shape=RMSNORM_HP, dtype="fp16",
                make_inputs=lambda s, d, dev: {
                    "x": torch.randn(*s, device=dev, dtype=d),
                    "r": torch.randn(*s, device=dev, dtype=d),
                    "w": torch.randn(s[-1], device=dev, dtype=d),
                },
                cuda_call=lambda mod, inp: mod.two_step_rmsnorm_residual(
                    inp["x"], inp["r"], inp["w"], 1e-6),
                ref_call=lambda inp: ref_fused_rmsnorm_residual(
                    inp["x"], inp["r"], inp["w"]),
                # 分步且 h 读两遍：读 x + 读 r + 写 h + 读 h + 读 h + 写 out = 6N
                nbytes=lambda s, d: 6 * workloads.numel(s) * workloads.elem_bytes(d),
                flops=lambda s, d: 7.0 * workloads.numel(s),
            ),
        ],
    ),
    KernelSpec(
        name="reduce",
        cuda_file="kernels/w2_reduce.cu",
        cpp_signature=("torch::Tensor reduce_sum(torch::Tensor x);"),
        functions=["reduce_sum"],
        note="W2:归约练习——树形归约 + 合并访存",
        cases=[
            Case(
                label= "(4194304,) fp32",
                shape=REDUCE_HP,
                dtype="fp32",
                make_inputs=lambda s, d, dev:{
                    "x": torch.rand(*s, device=dev, dtype=d),
                },
                cuda_call = lambda mod, inp: mod.reduce_sum(inp["x"]),
                ref_call = lambda inp:      ref_reduce(inp["x"]),
                nbytes=lambda s, d: workloads.bytes_elementwise(s, d, 1, 0)
            ),
        ],
    ),
    KernelSpec(
        name="attn_naive",
        cuda_file="kernels/w4_attn_naive.cu",
        cpp_signature=(
            "torch::Tensor attn_naive("
            "torch::Tensor Q, torch::Tensor K, torch::Tensor V);"
        ),
        functions=["attn_naive"],
        note="W4：naive attention（QKᵀ → softmax → PV 三个 kernel，S² 落显存）。"
             "基线：预期 4~6 ms、带宽 ~20%，用来证明 memory-bound + 给 w5 flash 当分母",
        cases=[
            Case(
                label="(4,8,1024,64) fp16",
                shape=(4, 8, 1024, 64),
                dtype="fp16",
                make_inputs=lambda s, d, dev: {
                    "Q": torch.randn(*s, device=dev, dtype=d),
                    "K": torch.randn(*s, device=dev, dtype=d),
                    "V": torch.randn(*s, device=dev, dtype=d),
                },
                cuda_call=lambda mod, inp: mod.attn_naive(
                    inp["Q"], inp["K"], inp["V"]),
                ref_call=lambda inp: ref_attn_naive(
                    inp["Q"], inp["K"], inp["V"]),
                # naive 的流量：Q/K/V/O 各 S*D（4 个），S_mat/P 的 4 次往返各 S*S
                #   → 每个 (b,h) 共 4*S*D + 4*S*S 个元素
                # 注意这是「算法流量」，实测用 ncu 的 dram__bytes 对照
                nbytes=lambda s, d: (
                    (4 * s[2] * s[3] + 4 * s[2] * s[2]) * s[0] * s[1]
                    * workloads.elem_bytes(d)
                ),
                # QKᵀ: 2*S²*D，PV: 2*S²*D，每个 (b,h)
                flops=lambda s, d: 4.0 * s[0] * s[1] * s[2] * s[2] * s[3],
            ),
        ],
    ),
    KernelSpec(
        name="gemm",
        cuda_file="kernels/w6_gemm.cu",
        cpp_signature="torch::Tensor gemm(torch::Tensor A, torch::Tensor B);",
        functions=["gemm"],
        note="W6：GEMM 台阶1（naive，每线程算 1 个 C 元素，直读全局内存）。"
             "基线预期 ~19 ms / ~110 GFLOPS；台阶2 加 shared tiling 后应掉到亚毫秒",
        cases=[
            Case(
                label="M=N=K=1024 fp16",
                shape=(1024, 1024, 1024),
                dtype="fp16",
                make_inputs=lambda s, d, dev: {
                    "A": torch.randn(s[0], s[1], device=dev, dtype=d),
                    "B": torch.randn(s[1], s[2], device=dev, dtype=d),
                },
                cuda_call=lambda mod, inp: mod.gemm(inp["A"], inp["B"]),
                ref_call=lambda inp: ref_gemm(inp["A"], inp["B"]),
                # GEMM 是 compute-bound，带宽口径只作"最小流量"参考：
                # A、B、C 各读/写一次
                nbytes=lambda s, d: (
                    (s[0] * s[1] + s[1] * s[2] + s[0] * s[2])
                    * workloads.elem_bytes(d)
                ),
                flops=lambda s, d: 2.0 * s[0] * s[1] * s[2],
            ),
        ],
    ),

        KernelSpec(
        name="gemm_tiled",
        cuda_file="kernels/w6_gemm.cu",
        cpp_signature="torch::Tensor gemm_tiled(torch::Tensor A, torch::Tensor B);",
        functions=["gemm_tiled"],
        note="W6：GEMM 台阶2（shared tiling 32×32，每线程算 1 个 C 元素；共享内存 [32][32] 无 padding，曾试 [32][33] 反而慢 41%）。",
        cases=[
            Case(
                label="M=N=K=1024 fp16",
                shape=(1024, 1024, 1024),
                dtype="fp16",
                make_inputs=lambda s, d, dev: {
                    "A": torch.randn(s[0], s[1], device=dev, dtype=d),
                    "B": torch.randn(s[1], s[2], device=dev, dtype=d),
                },
                cuda_call=lambda mod, inp: mod.gemm_tiled(inp["A"], inp["B"]),
                ref_call=lambda inp: ref_gemm(inp["A"], inp["B"]),
                nbytes=lambda s, d: (
                    (s[0] * s[1] + s[1] * s[2] + s[0] * s[2])
                    * workloads.elem_bytes(d)
                ),
                flops=lambda s, d: 2.0 * s[0] * s[1] * s[2],
            ),
        ],
    ),
    KernelSpec(
        name="gemm_reg",
        cuda_file="kernels/w6_gemm.cu",
        cpp_signature="torch::Tensor gemm_reg(torch::Tensor A, torch::Tensor B);",
        functions=["gemm_reg"],
        note="W6：GEMM 台阶3（寄存器分块 4×4，block 8×8=64 线程；"
             "每线程算 16 个 C 元素，共享内存读取摊薄 4 倍（每读服务 4 次 FMA））。"
             "台阶4 再上 Tensor Core",
        cases=[
            Case(
                label="M=N=K=1024 fp16",
                shape=(1024, 1024, 1024),
                dtype="fp16",
                make_inputs=lambda s, d, dev: {
                    "A": torch.randn(s[0], s[1], device=dev, dtype=d),
                    "B": torch.randn(s[1], s[2], device=dev, dtype=d),
                },
                cuda_call=lambda mod, inp: mod.gemm_reg(inp["A"], inp["B"]),
                ref_call=lambda inp: ref_gemm(inp["A"], inp["B"]),
                nbytes=lambda s, d: (
                    (s[0] * s[1] + s[1] * s[2] + s[0] * s[2])
                    * workloads.elem_bytes(d)
                ),
                flops=lambda s, d: 2.0 * s[0] * s[1] * s[2],
            ),
        ],
    ),
    KernelSpec(
        name="gemm_tc",
        cuda_file="kernels/w6_gemm.cu",
        cpp_signature="torch::Tensor gemm_tc(torch::Tensor A, torch::Tensor B);",
        functions=["gemm_tc"],
        note="W6：GEMM 台阶4（Tensor Core / wmma 16×16×16，block 32×4=128 线程=4 warp，"
             "每个 warp 算 C tile 的一个 16×16 子块；fp32 累加后转 fp16 写回）。"
             "参考实现仍是 PyTorch fp32 matmul，所以台阶4 可以直接越过基线",
        cases=[
            Case(
                label="M=N=K=1024 fp16",
                shape=(1024, 1024, 1024),
                dtype="fp16",
                make_inputs=lambda s, d, dev: {
                    "A": torch.randn(s[0], s[1], device=dev, dtype=d),
                    "B": torch.randn(s[1], s[2], device=dev, dtype=d),
                },
                cuda_call=lambda mod, inp: mod.gemm_tc(inp["A"], inp["B"]),
                ref_call=lambda inp: ref_gemm(inp["A"], inp["B"]),
                nbytes=lambda s, d: (
                    (s[0] * s[1] + s[1] * s[2] + s[0] * s[2])
                    * workloads.elem_bytes(d)
                ),
                flops=lambda s, d: 2.0 * s[0] * s[1] * s[2],
            ),
        ],
    ),
    KernelSpec(
        name="gemm_tc2",
        cuda_file="kernels/w6_gemm.cu",
        cpp_signature="torch::Tensor gemm_tc2(torch::Tensor A, torch::Tensor B);",
        functions=["gemm_tc2"],
        note="W6：GEMM 台阶4b（Tensor Core + 64×64 tile，block 32×16=512 线程=16 warp，"
             "每 warp 一个 16×16 子块；As[64][32] / Bs[32][64] / Cf[64][64]，共享内存 24 KB）。"
             "相对台阶4 的 32×32 tile：warp 数 4→16，每 SM 驻留 3 个 block",
        cases=[
            Case(
                label="M=N=K=1024 fp16",
                shape=(1024, 1024, 1024),
                dtype="fp16",
                make_inputs=lambda s, d, dev: {
                    "A": torch.randn(s[0], s[1], device=dev, dtype=d),
                    "B": torch.randn(s[1], s[2], device=dev, dtype=d),
                },
                cuda_call=lambda mod, inp: mod.gemm_tc2(inp["A"], inp["B"]),
                ref_call=lambda inp: ref_gemm(inp["A"], inp["B"]),
                nbytes=lambda s, d: (
                    (s[0] * s[1] + s[1] * s[2] + s[0] * s[2])
                    * workloads.elem_bytes(d)
                ),
                flops=lambda s, d: 2.0 * s[0] * s[1] * s[2],
            ),
        ],
    ),
    KernelSpec(
        name="paged_attn",
        cuda_file="kernels/w7_paged_attn.cu",
        cpp_signature=(
            "torch::Tensor paged_attn("
            "torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache, "
            "torch::Tensor block_table, torch::Tensor context_lens);"
        ),
        functions=["paged_attn"],
        note="W7：PagedAttention decode（KV cache 分块 + 块表间接寻址，块表乱序）。"
             "算术强度 ≈1 -> 纯 memory-bound，decode 的典型特征",
        cases=[
            Case(
                label="(8,32,2048,128) fp16",
                shape=(8, 32, 2048, 128),
                dtype="fp16",
                make_inputs=_make_paged_inputs,
                cuda_call=lambda mod, inp: mod.paged_attn(
                    inp["q"], inp["k_cache"], inp["v_cache"],
                    inp["block_table"], inp["context_lens"]),
                ref_call=lambda inp: ref_paged_attn(
                    inp["q"], inp["k_cache"], inp["v_cache"],
                    inp["block_table"], inp["context_lens"]),
                # 流量：q + out（各 B*H*D）+ K/V 各 B*H*ctx*D（每个 (b,h) 都完整读一遍）
                nbytes=lambda s, d: (
                    (2 * s[0] * s[1] * s[3] + 2 * s[0] * s[1] * s[2] * s[3])
                    * workloads.elem_bytes(d)
                ),
                # QK: 2*B*H*ctx*D，PV: 2*B*H*ctx*D
                flops=lambda s, d: 4.0 * s[0] * s[1] * s[2] * s[3],
            ),
        ],
    ),

    KernelSpec(
        name="paged_attn_split",
        cuda_file="kernels/w7_paged_attn_split.cu",
        cpp_signature=(
            "torch::Tensor paged_attn_split("
            "torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache, "
            "torch::Tensor block_table, torch::Tensor context_lens);"
        ),
        functions=["paged_attn_split"],
        note="W7b：split-L 版（把 L 切 4 段并行，再用 merge kernel 按 online softmax 合并）。"
             "工作单元 B*H*4=1024，目标是吃满带宽",
        cases=[
            Case(
                label="(8,32,2048,128) fp16",
                shape=(8, 32, 2048, 128),
                dtype="fp16",
                make_inputs=_make_paged_inputs,
                cuda_call=lambda mod, inp: mod.paged_attn_split(
                    inp["q"], inp["k_cache"], inp["v_cache"],
                    inp["block_table"], inp["context_lens"]),
                ref_call=lambda inp: ref_paged_attn(
                    inp["q"], inp["k_cache"], inp["v_cache"],
                    inp["block_table"], inp["context_lens"]),
                nbytes=lambda s, d: (
                    (2 * s[0] * s[1] * s[3] + 2 * s[0] * s[1] * s[2] * s[3])
                    * workloads.elem_bytes(d)
                ),
                flops=lambda s, d: 4.0 * s[0] * s[1] * s[2] * s[3],
            ),
        ],
    ),
    KernelSpec(
        name="paged_attn_int8",
        cuda_file="kernels/w8_paged_attn_int8.cu",
        cpp_signature=(
            "torch::Tensor paged_attn_int8("
            "torch::Tensor q, torch::Tensor k_q, torch::Tensor k_scale, "
            "torch::Tensor v_cache, torch::Tensor block_table, "
            "torch::Tensor context_lens);"
        ),
        functions=["paged_attn_int8"],
        note="W8：PagedAttention + INT8 KV cache（台阶1：K 存 int8、读出来反量化）。"
             "K 流量 128->64 MiB，总流量 256->193 MiB",
        cases=[
            Case(
                label="(8,32,2048,128) fp16q/int8k",
                shape=(8, 32, 2048, 128),
                dtype="fp16",
                make_inputs=_make_paged_int8_inputs,
                cuda_call=lambda mod, inp: mod.paged_attn_int8(
                    inp["q"], inp["k_q"], inp["k_scale"], inp["v_cache"],
                    inp["block_table"], inp["context_lens"]),
                ref_call=lambda inp: ref_paged_attn_int8(
                    inp["q"], inp["k_q"], inp["k_scale"], inp["v_cache"],
                    inp["block_table"], inp["context_lens"]),
                # q/out 2*B*H*D*2B + K B*H*ctx*D*1B + scale B*H*ctx*2B + V B*H*ctx*D*2B
                nbytes=lambda s, d: (
                    2 * s[0] * s[1] * s[3] * 2
                    + s[0] * s[1] * s[2] * s[3] * 1
                    + s[0] * s[1] * s[2] * 2
                    + s[0] * s[1] * s[2] * s[3] * 2
                ),
                flops=lambda s, d: 4.0 * s[0] * s[1] * s[2] * s[3],
            ),
        ],
    ),
    KernelSpec(
        name="paged_attn_int8kv",
        cuda_file="kernels/w8_paged_attn_int8kv.cu",
        cpp_signature=(
            "torch::Tensor paged_attn_int8kv("
            "torch::Tensor q, torch::Tensor k_q, torch::Tensor k_scale, "
            "torch::Tensor v_q, torch::Tensor v_scale, "
            "torch::Tensor block_table, torch::Tensor context_lens);"
        ),
        functions=["paged_attn_int8kv"],
        note="W8 台阶2：K/V 都 int8（V 的 scale 折进 p）。流量 256 -> 137.6 MiB",
        cases=[
            Case(
                label="(8,32,2048,128) int8k+int8v",
                shape=(8, 32, 2048, 128),
                dtype="fp16",
                make_inputs=_make_paged_int8kv_inputs,
                cuda_call=lambda mod, inp: mod.paged_attn_int8kv(
                    inp["q"], inp["k_q"], inp["k_scale"], inp["v_q"], inp["v_scale"],
                    inp["block_table"], inp["context_lens"]),
                ref_call=lambda inp: ref_paged_attn_int8kv(
                    inp["q"], inp["k_q"], inp["k_scale"], inp["v_q"], inp["v_scale"],
                    inp["block_table"], inp["context_lens"]),
                # q/out 2*B*H*D*2B + K,V 各 B*H*ctx*D*1B + 两个 scale 各 B*H*ctx*2B
                nbytes=lambda s, d: (
                    2 * s[0] * s[1] * s[3] * 2
                    + 2 * s[0] * s[1] * s[2] * s[3] * 1
                    + 2 * s[0] * s[1] * s[2] * 2
                ),
                flops=lambda s, d: 4.0 * s[0] * s[1] * s[2] * s[3],
            ),
        ],
    ),
]


def ensure_ninja_on_path():
    """确保 `ninja` **可执行文件**在 PATH 上。

    为什么需要这一步（踩过的坑）：
        `load_inline` 是通过 subprocess 调用 `ninja` 可执行文件的，
        不是 `import ninja`。所以 `pip install ninja` 装了 Python 包还不够 ——
        可执行文件落在 `<环境>/bin/ninja`，**只有激活了那个环境、该目录在 PATH 上**
        才找得到。

        症状：明明 `pip install ninja` 成功、`import ninja` 也没问题，
        但 load_inline 仍然报
            RuntimeError: Ninja is required to load C++ extensions

    这里做一个定向修复：如果 PATH 上没有 ninja，就看看它是否就在当前
    Python 解释器旁边（即当前环境的 bin/ 目录），是的话把它加进 PATH。
    这样即使忘了 `conda activate` 也能跑。

    返回 ninja 路径；找不到返回 None。
    """
    found = shutil.which("ninja")
    if found:
        return found

    env_bin = Path(sys.executable).parent
    cand = env_bin / "ninja"
    if cand.is_file() and os.access(cand, os.X_OK):
        os.environ["PATH"] = str(env_bin) + os.pathsep + os.environ.get("PATH", "")
        return str(cand)

    # 再兜底找几个常见位置
    for p in ("/usr/bin/ninja", "/usr/local/bin/ninja", str(Path.home() / ".local/bin/ninja")):
        if os.path.isfile(p) and os.access(p, os.X_OK):
            os.environ["PATH"] = str(Path(p).parent) + os.pathsep + os.environ.get("PATH", "")
            return p
    return None


# ---------------------------------------------------------------------------
# 内置自检 kernel —— 用来验证「CUDA 编译链路」通不通。**这不是你的作业。**
# ---------------------------------------------------------------------------

HARNESS_CUDA = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void scale_kernel(const float* __restrict__ in,
                             float* __restrict__ out,
                             const float factor, const long long n) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] * factor;
}

torch::Tensor scale(torch::Tensor x, double factor) {
    TORCH_CHECK(x.is_cuda(), "x 必须在 GPU 上");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "自检只支持 fp32");
    auto xc = x.contiguous();
    auto out = torch::empty_like(xc);
    const long long n = (long long)xc.numel();
    const int threads = 256;
    const long long blocks = (n + threads - 1) / threads;
    scale_kernel<<<(unsigned int)blocks, threads>>>(
        xc.data_ptr<float>(), out.data_ptr<float>(), (float)factor, n);
    return out;
}
"""

_HARNESS_N = 1 << 22  # 4M 个 fp32 元素


def harness_spec() -> KernelSpec:
    n = _HARNESS_N
    return KernelSpec(
        name="_harness_selftest",
        cuda_file="(内置源码)",
        cpp_signature="torch::Tensor scale(torch::Tensor x, double factor);",
        functions=["scale"],
        note="内置自检：验证 nvcc → load_inline → 正确性校验 → 计时 整条链路",
        cuda_source=HARNESS_CUDA,
        cases=[Case(
            label=f"({n},) fp32",
            shape=(n,),
            dtype="fp32",
            make_inputs=lambda s, d, dev: {"x": torch.randn(*s, device=dev, dtype=d)},
            cuda_call=lambda mod, inp: mod.scale(inp["x"], 2.0),
            ref_call=lambda inp: inp["x"] * 2.0,
            nbytes=lambda s, d: 2 * workloads.numel(s) * workloads.elem_bytes(d),
            flops=lambda s, d: float(workloads.numel(s)),
        )],
    )


# ---------------------------------------------------------------------------
# 编译与测量
# ---------------------------------------------------------------------------

def build_module(spec: KernelSpec, verbose: bool):
    """编译一个 kernel。返回 (module, error_message)。"""
    from torch.utils.cpp_extension import load_inline

    if spec.cuda_source:
        src = spec.cuda_source
    else:
        path = Path(spec.cuda_file)
        if not path.exists():
            return None, f"文件不存在：{spec.cuda_file}（还没写）"
        src = path.read_text(encoding="utf-8")

    try:
        mod = load_inline(
            name=f"k_{spec.name}",
            cpp_sources=spec.cpp_signature,
            cuda_sources=src,
            functions=spec.functions,
            extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
            verbose=verbose,
        )
        return mod, None
    except Exception as e:  # noqa: BLE001
        if verbose:
            traceback.print_exc()
        msg = str(e).strip().splitlines()
        return None, f"编译失败：{msg[-1] if msg else type(e).__name__}"


def measure_case(mod, spec: KernelSpec, case: Case, args) -> dict:
    dtype = DTYPES[case.dtype]
    row = dict(kernel=spec.name, case=case.label, status="", ms=None, ref_ms=None,
               speedup=None, mem=None, bw=None, bw_util=None, tflops=None,
               bound="-", err="")

    try:
        inp = case.make_inputs(case.shape, dtype, DEV)
    except Exception as e:  # noqa: BLE001
        row["status"], row["err"] = "FAIL", f"构造输入失败：{e}"
        return row

    try:
        cand = case.cuda_call(mod, inp)
    except Exception as e:  # noqa: BLE001
        row["status"], row["err"] = "FAIL", f"调用失败：{type(e).__name__}: {e}"
        return row

    ref = case.ref_call(inp)
    chk = verify.check(ref, cand, name=f"{spec.name}/{case.label}", verbose=False)
    row["status"] = "PASS" if chk["ok"] else "WRONG"
    row["err"] = ("" if chk["ok"] else
                  f"max_abs={chk['max_abs_err']:.2e} cos={chk['cosine_sim']:.6f}")
    if not chk["ok"]:
        print(f"  [WRONG] {case.label}  {row['err']}")
        return row

    # 峰值显存
    try:
        timer.release_flush_buffers()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        case.cuda_call(mod, inp)
        torch.cuda.synchronize()
        row["mem"] = torch.cuda.max_memory_allocated() / 1024 ** 2
    except Exception:  # noqa: BLE001
        pass

    try:
        st = timer.bench(lambda: case.cuda_call(mod, inp),
                         warmup=args.warmup, iters=args.iters,
                         flush_l2=not args.no_flush_l2)
    except Exception as e:  # noqa: BLE001
        torch.cuda.empty_cache()
        row["status"], row["err"] = "FAIL", f"计时失败：{type(e).__name__}: {e}"
        print(f"  [FAIL] {case.label}  {row['err']}")
        return row
    row["ms"] = st["median_ms"]

    try:
        st_ref = timer.bench(lambda: case.ref_call(inp),
                             warmup=args.warmup, iters=args.iters,
                             flush_l2=not args.no_flush_l2)
        row["ref_ms"] = st_ref["median_ms"]
        if row["ms"] and row["ms"] > 0:
            row["speedup"] = row["ref_ms"] / row["ms"]
    except Exception:  # noqa: BLE001
        pass

    nb = case.nbytes(case.shape, dtype)
    row["bw"] = workloads.bandwidth_gbps(nb, row["ms"])
    row["bw_util"] = 100 * row["bw"] / args.spec.bw_gbps
    if case.flops:
        fl = case.flops(case.shape, dtype)
        row["tflops"] = workloads.achieved_tflops(fl, row["ms"])
        row["bound"] = args.spec.classify(fl, nb, case.dtype)

    speed = f"  加速 {row['speedup']:.2f}x" if row["speedup"] else ""
    print(f"  [PASS]  {case.label:<22} {row['ms']:8.4f} ms  "
          f"带宽 {row['bw']:7.1f} GB/s ({row['bw_util']:5.1f}%){speed}")
    timer.release_flush_buffers()
    return row


def run_self_test(args) -> int:
    """不写任何 CUDA，直接测参考实现 —— 用来确认 harness 本身能跑。"""
    print("=" * 74)
    print("自检模式：只测 PyTorch 参考实现，不需要你写 kernel")
    print("=" * 74)
    rows = []
    for spec in REGISTRY:
        for case in spec.cases:
            dtype = DTYPES[case.dtype]
            inp = case.make_inputs(case.shape, dtype, DEV)
            st = timer.bench(lambda: case.ref_call(inp),
                             warmup=args.warmup, iters=args.iters,
                             flush_l2=not args.no_flush_l2)
            nb = case.nbytes(case.shape, dtype)
            bw = workloads.bandwidth_gbps(nb, st["median_ms"])
            rows.append([spec.name, case.label, st["median_ms"], bw,
                         100 * bw / args.spec.bw_gbps])
            print(f"  {spec.name:<26} {st['median_ms']:8.4f} ms   "
                  f"带宽 {bw:7.1f} GB/s ({100*bw/args.spec.bw_gbps:5.1f}%)")
            timer.release_flush_buffers()

    print()
    print(report.md_table(["kernel", "case", "PyTorch 参考(ms)",
                           "有效带宽(GB/s)", "带宽利用率(%)"], rows))
    print("\n如果上面有数字，说明 harness 正常。现在去写你的 kernel 吧。")
    print("写完把文件放到 kernels/ 下、文件名对上，再跑不带 --self-test 的版本。")
    return 0


def run_harness_test(args) -> int:
    """编译一个内置的平凡 kernel，验证整条 CUDA 编译链路。"""
    print("=" * 74)
    print("编译链路自检：编译一个内置的平凡 kernel（scale）")
    print("它**不是你的作业**，只用来确认工具链能用。")
    print("=" * 74)

    spec_item = harness_spec()
    mod, err = build_module(spec_item, verbose=True)
    if mod is None:
        print(f"\n[x] 编译链路有问题：{err}")
        print("\n按顺序排查：")
        print("  0. ninja 是否安装     : pip install ninja"
              "        ← load_inline 的构建后端，最容易漏")
        print("  1. nvcc 是否可用      : nvcc --version")
        print("  2. 计算能力是否匹配   : export TORCH_CUDA_ARCH_LIST='8.9'  # 你的卡是 SM 8.9")
        print("  3. 磁盘空间是否够     : df -h ~   （编译缓存默认在 ~/.cache/torch）")
        print("  4. 编译缓存损坏       : rm -rf ~/.cache/torch_extensions")
        return 1

    print("\n[+] 编译成功。开始验证正确性并计时 ...")
    r = measure_case(mod, spec_item, spec_item.cases[0], args)
    if r["status"] != "PASS":
        print(f"\n[x] 编译过了，但测量环节有问题：{r['err']}")
        return 1
    print("\n[+] 整条链路正常（编译 → 正确性 → 计时 → 带宽计算）。")
    print("    现在可以放心写你的 kernel 了。")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="编译、验证、测量你的 CUDA kernel")
    ap.add_argument("--only", default=None, help="只测名字包含该子串的 kernel")
    ap.add_argument("--self-test", action="store_true",
                    help="只测 PyTorch 参考实现，不编译 CUDA")
    ap.add_argument("--harness-test", action="store_true",
                    help="编译一个内置的平凡 kernel，验证 CUDA 编译链路能否跑通")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--no-flush-l2", action="store_true")
    ap.add_argument("--verbose-build", action="store_true", help="显示编译全过程")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA 不可用。先跑：python check_env.py", file=sys.stderr)
        return 2

    nj = ensure_ninja_on_path()
    if nj is None:
        print("[!] PATH 上找不到 ninja 可执行文件 —— 第一次编译 kernel 会失败。", file=sys.stderr)
        print("    修复：conda activate <你的环境>   然后重开终端", file=sys.stderr)
        print("    验证：which ninja", file=sys.stderr)
    else:
        print(f"ninja: {nj}")

    args.spec = specs.detect()
    print("=" * 74)
    print(args.spec)
    print("=" * 74)

    if args.self_test:
        return run_self_test(args)
    if args.harness_test:
        return run_harness_test(args)

    selected = [s for s in REGISTRY
                if args.only is None or args.only in s.name]

    rows, points = [], []
    for spec_item in selected:
        print(f"\n--- {spec_item.name} ---")
        print(f"    {spec_item.note}")
        mod, err = build_module(spec_item, args.verbose_build)
        if mod is None:
            print(f"  跳过：{err}")
            rows.append([spec_item.name, "-", "SKIP", None, None, None,
                         None, None, None, err])
            continue

        for case in spec_item.cases:
            r = measure_case(mod, spec_item, case, args)
            rows.append([r["kernel"], r["case"], r["status"], r["ms"], r["ref_ms"],
                         r["speedup"], r["mem"], r["bw_util"], r["bound"], r["err"]])
            if r["status"] == "PASS" and r["tflops"]:
                fl = case.flops(case.shape, DTYPES[case.dtype])
                nb = case.nbytes(case.shape, DTYPES[case.dtype])
                points.append(dict(name=f"{r['kernel']} ({case.label})",
                                   ai=fl / nb, tflops=r["tflops"]))
        del mod
        torch.cuda.empty_cache()

    table = report.md_table(
        ["kernel", "case", "状态", "我的(ms)", "PyTorch(ms)", "加速比",
         "峰值显存(MB)", "带宽利用率(%)", "Roofline", "备注"], rows)

    out = Path(args.out_dir)
    sections = [
        report.env_block(args.spec),
        "## 结果\n", table, "",
        "> 状态含义：`PASS` 正确且已测；`WRONG` 结果不对（先修正确性再谈性能）；\n"
        "> `SKIP` 文件还没写；`FAIL` 调用/编译出错。\n",
        "## W2/W3 验收标准\n",
        "- W2：至少一个算子**带宽利用率 > 70%**（即 > 190 GB/s），且最大相对误差 < 1e-3\n"
        "- W3：融合算子相对分步实现有明确加速比（memory-bound 场景通常 1.5x–3x），\n"
        "  并能算出节省了多少次 HBM 往返\n",
        "## 每个算子都要能回答的问题\n",
        "1. `block_size` 为什么是这个值？改成 128 或 512 性能怎么变？（**测出来**）\n"
        "2. 用了什么规约策略？warp shuffle / 共享内存 / 树形规约？\n"
        "3. 共享内存用了多少字节？occupancy 是多少？\n"
        "4. 有效带宽到了峰值的百分之多少？\n"
        "5. 有没有 bank conflict / warp divergence？怎么验证的？\n",
    ]
    report.write_report(out / "kernels.md", "自写 CUDA Kernel 性能报告", sections)

    if points:
        plotting.plot_roofline(args.spec, points,
                               out / "fig_roofline_kernels.png",
                               dtype="fp16",
                               title=f"Hand-written kernels on {args.spec.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
