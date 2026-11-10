# -*- coding: utf-8 -*-
"""M5：continuous batching（decode 批处理）

要解决的问题
------------
M4 里每个 decode 步只服务 1 个请求：B=1, H=12, NSPLIT=4 → 48 个 warp = 1536 线程
= 6 个 block（256 线程/block），而卡上有 24 个 SM ⇒ **75% 的 SM 闲着**。

M5 把 8 个请求的「最后一个 token」拼成一个 batch，一次 forward 产出 8 个 token：
48 warp → 384 warp = 48 个 block，24 个 SM 每个 2 个。

三个动作
--------
1. 请求状态机：waiting / running / finished，每个请求自己的 block_table 与 n
2. 每步组 batch：把 running 里每个请求的最后一个 token 拼成 [B, 1]
3. 采样 + 回收：每请求各自 argmax；结束后把块还回 free

与 M3/M4 的关键差别
-------------------
- **池与分配器改成全局共享**（M3/M4 是「每请求一个池」，8 个请求会占 8×151 MB）；
  这是 kernel 能一次算 B 个请求的前提 —— 它只接受**一个** k_cache/v_cache。
- `paged_attn_split` **不用改**：它的 block_table 是 [B, MAXB]、context_lens 是 [B]、
  grid 是 B*H*NSPLIT，本来就支持 B>1。

判据（推导见 docs/06-项目/01-mini推理引擎.md §13）
-----------------------------------------------
- 吞吐 ≥ 串行 × 2（目标 4×；硬上限 8×，超过说明测错了）
- 逐序列 token 与串行基线完全一致；不一致时打印 logits 误差帮助定位
- 跑完 free 块数回到 N_BLOCKS（不泄漏）

明确不做：chunked prefill（需重写 kernel 支持 S>1）、preemption、TP

跑：
    HF_HUB_OFFLINE=1 ~/miniconda3/envs/triton/bin/python engines/m5_batching.py
    # 第一次会编译 CUDA（1-2 分钟）
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import pathlib
import statistics as st
import sys
import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
from transformers import AutoModelForCausalLM, AutoTokenizer

from m3_paged import BLOCK_SIZE, KVPool, BlockAllocator
from m3_paged_attn import PagedState

MODEL = "gpt2"
DTYPE = torch.float16
STEPS = 20                       # 每个请求生成多少个 token
ROUNDS = 3                       # 交替测量轮数（丢第 1 轮）
MAX_BATCH = 8
N_BLOCKS_M5 = 256                # 全局共享池：256 块 × 16 = 4096 个 token 容量
KERNEL = pathlib.Path(__file__).resolve().parent.parent / "kernels" / "w7_paged_attn_split.cu"

DECL = """torch::Tensor paged_attn_split(torch::Tensor q, torch::Tensor k_cache,
                                  torch::Tensor v_cache, torch::Tensor block_table,
                                  torch::Tensor context_lens);"""

# 8 个长度不同的 prompt —— 长度不同才能证明「每请求一张表」是对的
PROMPTS = [
    "The history of the Roman Empire is long and complicated. ",
    "In 1492, Columbus sailed across the Atlantic Ocean. ",
    "The quick brown fox jumps over the lazy dog. ",
    "Machine learning models require large amounts of data. ",
    "The capital of France is Paris and the river is the Seine. ",
    "Quantum computers use qubits instead of classical bits. ",
    "She opened the door and walked into the dark room. ",
    "The recipe calls for two cups of flour and one egg. ",
]


# ---------------------------------------------------------------- kernel
def build_kernel():
    return load_inline(
        name="m5_paged_split",
        cpp_sources=DECL,
        cuda_sources=KERNEL.read_text(encoding="utf-8"),
        functions=["paged_attn_split"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )


# ---------------------------------------------------------------- 请求
@dataclass
class Request:
    """一个在跑的请求：自己的表、自己的 n、自己生成的 token"""
    rid: int
    prompt: torch.Tensor                    # [1, S]
    state: PagedState
    tokens: list = field(default_factory=list)
    logits: torch.Tensor = None             # 最后一步的 logits，用来比对

    @property
    def n_done(self):
        return len(self.tokens)


# ---------------------------------------------------------------- 接线
class BatchPatcher:
    """把 12 层的 attention 换成我们的实现，支持 batch 内多个请求

    与 m3_paged_kernel.AttnPatcher 的差别：
        - `self.states` 是**列表**（每行一个请求），不是单个 state
        - `tab_t` 是 [B, MAXB]、`lens_t` 是 [B]（kernel 本来就吃这个形状）
        - decode 时一次 forward 算 B 个请求，prefill 时 B=1
    """

    def __init__(self, model, mod, pool, n_blocks):
        self.blocks = model.transformer.h if hasattr(model, "transformer") else model.h
        self.orig = [b.attn.forward for b in self.blocks]
        self.patched = [self._make(i, b.attn, mod, pool) for i, b in enumerate(self.blocks)]
        self.states = []                    # 本步参与 forward 的 PagedState 列表
        self.tab_t = None                   # [B, MAXB] int32
        self.lens_t = None                  # [B] int32
        self.n_blocks = n_blocks
        self.stats = {"kernel": 0, "sdpa": 0, "kernel_ms": 0.0}

    def _make(self, layer_idx, attn, mod, pool):
        H, D, scaling = attn.num_heads, attn.head_dim, attn.scaling

        def patched(hidden_states, past_key_values=None, attention_mask=None,
                    output_attentions=False, **kw):
            B, S, _ = hidden_states.shape
            qkv = attn.c_attn(hidden_states)
            q, k, v = qkv.split(attn.split_size, dim=2)
            q = q.view(B, S, H, D).transpose(1, 2)
            k = k.view(B, S, H, D).transpose(1, 2)
            v = v.view(B, S, H, D).transpose(1, 2)

            # 每一行 K/V 写进它自己请求的那张表
            assert B == len(self.states), f"batch={B} 与 states={len(self.states)} 不一致"
            for i, state in enumerate(self.states):
                state.append(layer_idx, k[i], v[i])

            if layer_idx == 0:
                for state in self.states:
                    state.n += S
                # 页表与长度每步只刷一次，12 层复用同一块显存
                self.tab_t.zero_()
                for i, state in enumerate(self.states):
                    self.tab_t[i, :len(state.table)] = torch.tensor(
                        state.table, dtype=torch.int32, device=q.device)
                    self.lens_t[i] = state.n

            if S > 1:                                   # prefill：单请求，SDPA
                K, V = self.states[0].gather(layer_idx)
                out = F.scaled_dot_product_attention(
                    q, K.unsqueeze(0), V.unsqueeze(0), is_causal=True, scale=scaling)
                self.stats["sdpa"] += 1
            else:                                       # decode：B 个请求一起走 kernel
                t0 = time.perf_counter()
                out = mod.paged_attn_split(
                    (q[:, :, 0, :] * scaling).contiguous(),
                    pool.k[layer_idx], pool.v[layer_idx],
                    self.tab_t, self.lens_t)             # [B, H, D]
                self.stats["kernel_ms"] += (time.perf_counter() - t0) * 1e3
                self.stats["kernel"] += 1
                out = out.unsqueeze(1)                   # [B, H, 1, D]

            out = out.transpose(1, 2).reshape(B, S, H * D)
            return attn.c_proj(out), None

        return patched

    def bind(self, states):
        """本步要服务哪些请求；同时按 batch 大小（重新）分配页表张量"""
        self.states = list(states)
        B = len(self.states)
        dev = next(self.blocks[0].parameters()).device
        if self.tab_t is None or self.tab_t.shape[0] < B:
            self.tab_t = torch.zeros(B, self.n_blocks, dtype=torch.int32, device=dev)
            self.lens_t = torch.zeros(B, dtype=torch.int32, device=dev)

    def use(self, on: bool):
        for i, blk in enumerate(self.blocks):
            blk.attn.forward = self.patched[i] if on else self.orig[i]


# ---------------------------------------------------------------- 批处理器（骨架）
class Batcher:
    """continuous batching 的调度器：waiting → running → finished"""

    def __init__(self, model, patcher, pool, alloc, max_batch=MAX_BATCH):
        self.model = model
        self.patcher = patcher
        self.pool = pool
        self.alloc = alloc
        self.max_batch = max_batch
        self.waiting = []               # list[torch.Tensor]，待进入的 prompt
        self.running = []               # list[Request]
        self.finished = []              # list[Request]
        self.peak_used = 0
        self.n_fwd = 0                  # 一共做了几次 forward（对比串行用）

    # ---------------- TODO 1 ----------------
    def admit(self):
        """把 waiting 里的请求拉进 running（新请求插进正在跑的 batch）

        要做的事：
          1. while self.waiting and len(self.running) < self.max_batch:
               看空闲块够不够（按 len(prompt) 算 needs = ceil(S/BS)，再留 2 块余量）；
               不够就 break（M5 不做 preemption，等别人跑完再说）
          2. 建一个 **共享池上的** PagedState：PagedState(n_layer, n_head, head_dim, DTYPE,
             pool=self.pool, alloc=self.alloc)
          3. patcher.bind([state]) 后用 model(ids, past_key_values=None, use_cache=False,
             position_ids=arange(0,S)) 跑 prefill，取 logits[:, -1, :].argmax(-1) 当第 1 个 token
          4. 装进 Request 放进 running

        注意：
          - 块的分配/增长在 state.append() 内部（layer 0）自动做，不用手算
          - self.n_fwd += 1（prefill 也是一次 forward）
        """
        raise NotImplementedError("TODO 1：实现 admit")

    # ---------------- TODO 2 ----------------
    def decode_step(self):
        """running 里所有请求各推 1 个 token —— M5 的核心就这一句

        要做的事：
          1. inp = torch.cat([r.last_token for r in self.running], dim=0)  # [B,1]
          2. 每个请求的 position_ids 不同（各自的 state.n），拼成 [B,1]：
             pos = torch.tensor([[r.state.n] for r in self.running])
          3. patcher.bind([r.state for r in self.running])，一次 forward
          4. 每个请求各自 argmax，存进 r.tokens / r.logits；够 STEPS 的挪进 finished
          5. self.n_fwd += 1；记 self.peak_used = max(self.peak_used, self.alloc.n_used)

        注意：**不要**在这里 gather，decode 走的是 kernel 的按表寻址
        """
        raise NotImplementedError("TODO 2：实现 decode_step")

    # ---------------- TODO 3 ----------------
    def run(self, n_steps=STEPS):
        """调度循环：直到所有请求处理完

        要做的事：
          while self.waiting or self.running:
              self.admit()
              if self.running: self.decode_step()
          结束后把每个 finished 请求的块还回 free（self.alloc.free_blocks(r.state.table)）
          返回 {rid: tokens}
        """
        raise NotImplementedError("TODO 3：实现 run")


# ---------------------------------------------------------------- 基线（我写的）
@torch.inference_mode()
def serial_baseline(model, patcher, pool, alloc, prompts, n_tokens=STEPS):
    """串行：一个请求跑完再跑下一个（M4 的做法），返回 {rid: tokens}"""
    out = {}
    for rid, prompt in enumerate(prompts):
        state = PagedState(12, 12, 64, DTYPE, pool=pool, alloc=alloc)
        patcher.bind([state])
        o = model(prompt, past_key_values=None, use_cache=False,
                  position_ids=torch.arange(0, prompt.shape[1], device=prompt.device).unsqueeze(0))
        toks = [int(o.logits[0, -1].argmax())]
        for _ in range(n_tokens - 1):
            nxt = torch.tensor([[toks[-1]]], device=prompt.device)
            o = model(nxt, past_key_values=None, use_cache=False,
                      position_ids=torch.tensor([[state.n]], device=prompt.device))
            toks.append(int(o.logits[0, -1].argmax()))
        out[rid] = toks
        alloc.free_blocks(state.table)          # 请求结束，块还回去
    return out


# ---------------------------------------------------------------- 主程序（我写的）
def timed(fn):
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    out = fn()
    e.record()
    torch.cuda.synchronize()
    return out, s.elapsed_time(e)


def median(v):
    return st.median(v) if v else float("nan")


def main():
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE).eval().cuda()
    mod = build_kernel()

    n_layer = len(model.transformer.h)
    H, D = model.config.n_head, model.config.n_embd // model.config.n_head
    pool = KVPool(n_layer, H, D, DTYPE, n_blocks=N_BLOCKS_M5)
    alloc = BlockAllocator(N_BLOCKS_M5)
    patcher = BatchPatcher(model, mod, pool, N_BLOCKS_M5)

    prompts = [torch.tensor([tok.encode(p)], device="cuda") for p in PROMPTS]
    print(f"prompt 长度: {[p.shape[1] for p in prompts]}")
    print(f"batch = {MAX_BATCH}, 每请求 {STEPS} token, 池 {N_BLOCKS_M5} 块 (= {N_BLOCKS_M5 * BLOCK_SIZE} token)")

    # 先各跑一次拿 token（正确性判据不受轮次影响）
    patcher.use(True)
    base_tokens = serial_baseline(model, patcher, pool, alloc, prompts)
    batcher = Batcher(model, patcher, pool, alloc)
    batcher.waiting = list(prompts)
    bat_tokens = batcher.run(STEPS)
    patcher.use(False)

    # 交替测量：奇数轮先跑串行，偶数轮先跑批量，消掉「谁先跑谁吃亏」
    t_serial, t_batch, fwd = [], [], []
    for r in range(ROUNDS):
        patcher.use(True)
        if r % 2 == 0:
            _, ms1 = timed(lambda: serial_baseline(model, patcher, pool, alloc, prompts))
            b = Batcher(model, patcher, pool, alloc)
            b.waiting = list(prompts)
            _, ms2 = timed(lambda: b.run(STEPS))
        else:
            b = Batcher(model, patcher, pool, alloc)
            b.waiting = list(prompts)
            _, ms2 = timed(lambda: b.run(STEPS))
            _, ms1 = timed(lambda: serial_baseline(model, patcher, pool, alloc, prompts))
        patcher.use(False)
        fwd.append(b.n_fwd)
        if r == 0:
            print(f"  第 1 轮（丢弃，含预热）：串行 {ms1:.1f} ms / 批量 {ms2:.1f} ms")
            continue
        t_serial.append(ms1)
        t_batch.append(ms2)

    ms_s, ms_b = median(t_serial), median(t_batch)
    n_tok = MAX_BATCH * STEPS
    tput_s, tput_b = 1000 * n_tok / ms_s, 1000 * n_tok / ms_b
    speedup = ms_s / ms_b

    print()
    print(f"{'实现':<10}{'总耗时(ms)':>12}{'吞吐(tok/s)':>14}{'forward 次数':>14}")
    print(f"{'串行':<10}{ms_s:>12.1f}{tput_s:>14.1f}{MAX_BATCH * (STEPS + 1):>14}")
    print(f"{'批量':<10}{ms_b:>12.1f}{tput_b:>14.1f}{median(fwd):>14.0f}")
    print(f"加速比 = {speedup:.2f}×   (判据 ≥ 2×, 目标 4×, 上限 {MAX_BATCH}×)")

    ok_tok = sum(1 for i in range(MAX_BATCH) if base_tokens[i] == bat_tokens[i])
    leak = alloc.n_used
    print(f"token 一致: {ok_tok}/{MAX_BATCH} 请求")
    print(f"跑完占用块数: {leak}（应为 0，不泄漏）")
    print(f"峰值占用块数: {batcher.peak_used} / {N_BLOCKS_M5}")
    print()
    ok = (speedup >= 2.0) and (ok_tok == MAX_BATCH) and (leak == 0)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
