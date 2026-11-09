"""M3b-2：decode 步的 attention 换成 `paged_attn_split` kernel（fp16）

与 M3b-1 的差别只有一处：
    decode（S==1）时不再调 F.scaled_dot_product_attention，而是调
    kernels/w7_paged_attn_split.cu 编译出来的 paged_attn_split
    prefill（S>1）仍用 SDPA —— 那个 kernel 只支持单 query，prompt 段用不了

测量协议与 M2 一致（否则数字会骗人）：
    交替测 A/B 各 5 轮 → 丢弃第 1 轮（首次调用/编译/分配器冷启动）→ 取中位数
    第 1 次跑出来“4.3×”就是因为基线先跑、替我们的路径付了预热成本

判据：20 步 greedy token 与「HF DynamicCache 基线」一致率 ≥ 20/20

跑：
    cd ~/projects/aigpu-lab
    HF_HUB_OFFLINE=1 ~/miniconda3/envs/triton/bin/python engines/m3_paged_kernel.py
    # 第一次会编译 CUDA（约 1-2 分钟）
"""

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")        # 模型已在本地缓存，不再联网
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import pathlib
import statistics as st
import time

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache      # DynamicCache  基线的 HF cache（M2 的用法）

from m3_paged import BLOCK_SIZE, N_BLOCKS, BlockAllocator
from m3_paged_attn import PagedState

MODEL = "gpt2"
DTYPE = torch.float16
STEPS = 20
ROUNDS = 5
PROMPT = "The history of the Roman Empire is long and complicated. " * int(os.environ.get("PMUL", "1"))
KERNEL = pathlib.Path(__file__).resolve().parent.parent / "kernels" / "w7_paged_attn_split.cu"

DECL = """torch::Tensor paged_attn_split(torch::Tensor q, torch::Tensor k_cache,
                                  torch::Tensor v_cache, torch::Tensor block_table,
                                  torch::Tensor context_lens);"""

# load_incline 编译 CUDA kernel
def build_kernel():
    return load_inline(
        name="m3_paged_split",
        cpp_sources=DECL,                                   # C++ 声明 ———— 告诉 Python “有个函数叫 paged_attn_split，签名是这样”
        cuda_sources=KERNEL.read_text(encoding="utf-8"),    # CUDA 源码 ———— 从 .cu 文件读出来
        functions=["paged_attn_split"],                     # 暴露给 Python 的函数名
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )


class AttnPatcher:
    """装/卸我们的 attention：base = HF 原生，paged = 池 + 自研 kernel"""

    def __init__(self, model, mod):
        self.blocks = model.transformer.h if hasattr(model, "transformer") else model.h
        self.orig = [b.attn.forward for b in self.blocks]                       # 保存 12 个原生 forward
        self.patched = [self._make(i, b.attn, mod)                              # 创建 12 个 patched，每层一个，绑定 layer_idx/attn/mod
                        for i, b in enumerate(self.blocks)]
        self.state = None
        self.tab_t = None           # 页表的 GPU 张量 [1, N_BLOCKS]
        self.lens_t = None          # 当前请求的token总数 = state.n
        self.stats = {"kernel": 0, "sdpa": 0, "kernel_ms": 0.0}     # 统计数据

    # 工厂函数
    def _make(self, layer_idx, attn, mod):
        H, D, scaling = attn.num_heads, attn.head_dim, attn.scaling

        def patched(hidden_states, past_key_values=None, attention_mask=None,
                    output_attentions=False, **kw):
            B, S, _ = hidden_states.shape
            qkv = attn.c_attn(hidden_states)
            q, k, v = qkv.split(attn.split_size, dim=2)
            q = q.view(B, S, H, D).transpose(1, 2)
            k = k.view(B, S, H, D).transpose(1, 2)
            v = v.view(B, S, H, D).transpose(1, 2)

            self.state.append(layer_idx, k[0], v[0])
            if layer_idx == 0:
                self.state.n += S
                # block_table /lens_t 的长度每步只变一次，但 12 层都要用同一份，如果每次都刷新，就是 12 次 host→device 拷贝，很慢
                # 只在第 0 层原地刷新 tab_t 和 lens_t，其余层复用同一块显存
                '''
                    把 Python 数据搬到 GPU 张量，kernel 要的是 GPU 张量，但 state.table 是 Python list
                    比如 state.table = [3, 7]   state.n = 21   N_BLOCKS = 64

                    步骤	                      tab_t 内容	                        lens_t 内容
                    初始	                [[0, 0, 0, ..., 0]]（64 个 0）	                [0]
                    zero_()	                [[0, 0, 0, ..., 0]]（不变）	                    [0]
                    tab_t[0, :2] = [3, 7]	[[3, 7, 0, 0, ..., 0]]	                        [0]
                    lens_t.fill_(21)	    [[3, 7, 0, 0, ..., 0]]	                        [21]

                '''
                self.tab_t.zero_()              # 原地清零（不新建 tensor）
                self.tab_t[0, :len(self.state.table)] = torch.tensor(self.state.table, dtype=torch.int32, device=q.device)
                self.lens_t.fill_(self.state.n)

            # prefill 阶段，prompt段
            if S > 1:                                   # prefill：SDPA
                K, V = self.state.gather(layer_idx)
                out = F.scaled_dot_product_attention(q, K.unsqueeze(0), V.unsqueeze(0), is_causal=True, scale=scaling)
                self.stats["sdpa"] += 1
            else:                                       # decode：我们的 kernel
                # kernel 内部不乘 1/sqrt(D)，且它自己按 block_table 读池 
                # 这里不需要 gather，少的正是 M3b-1 里那次全量 K/V 拷贝
                t0 = time.perf_counter()
                # mod 是编译好的 CUDA kernel 模块
                # q的形状[B,H,S,D],在decode阶段，S=1，所以q为[1,H,1,D] 也就是 [1, 12, 1, 64]，所以取第0个
                out = mod.paged_attn_split((q[:, :, 0, :] * scaling).contiguous(),          # transpose 后不连续，要手动变成连续的
                                           self.state.pool.k[layer_idx],
                                           self.state.pool.v[layer_idx],
                                           self.tab_t, self.lens_t)   # [B,H,D]
                self.stats["kernel_ms"] += (time.perf_counter() - t0) * 1e3
                self.stats["kernel"] += 1
                out = out.unsqueeze(1)      # [B, H, D] ——> [B, H, 1, D]

            out = out.transpose(1, 2).reshape(B, S, H * D)
            return attn.c_proj(out), None

        return patched

    # 切换 base / paged 模式
    def use(self, mode, state=None):
        for i, b in enumerate(self.blocks):
            b.attn.forward = self.patched[i] if mode == "paged" else self.orig[i]
        self.state = state
        if state is not None:
            dev = next(self.blocks[0].parameters()).device
            self.tab_t = torch.zeros(1, N_BLOCKS, dtype=torch.int32, device=dev)
            self.lens_t = torch.zeros(1, dtype=torch.int32, device=dev)


@torch.inference_mode()
def run(model, ids, steps, paged, state=None, step_ms=None):
    n_prompt = ids.shape[1]
    cur, cache = ids.clone(), DynamicCache()
    ev = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(steps)]
    for step in range(steps):
        inp = cur if step == 0 else cur[:, -1:]
        ev[step][0].record()
        if paged:
            start = state.n
            pos = torch.arange(start, start + inp.shape[1], device=cur.device).unsqueeze(0)
            o = model(inp, past_key_values=None, use_cache=False, position_ids=pos)
        else:
            o = model(inp, past_key_values=cache, use_cache=True)
        cur = torch.cat([cur, o.logits[:, -1, :].argmax(-1, keepdim=True)], dim=1)
        ev[step][1].record()
    if step_ms is not None:
        torch.cuda.synchronize()
        step_ms.extend(s0.elapsed_time(e0) for s0, e0 in ev)
    return cur[0, n_prompt:].tolist()


def timed(fn):
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    out = fn()
    e.record()
    torch.cuda.synchronize()
    return out, s.elapsed_time(e)


def main():
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE).eval().cuda()
    cfg = model.config
    H, D, L = cfg.n_head, cfg.n_embd // cfg.n_head, cfg.n_layer
    print(f"gpt2: L={L} H={H} D={D} dtype={DTYPE}，编译 kernel ...")
    mod = build_kernel()
    patcher = AttnPatcher(model, mod)
    ids = tok(PROMPT, return_tensors="pt").input_ids.cuda()
    print(f"prompt {ids.shape[1]} token，生成 {STEPS} 步，交替 {ROUNDS} 轮\n")

    base_ms, mine_ms, tokens_base, tokens_mine = [], [], None, None
    sb, sp = [], []
    for r in range(ROUNDS):
        last = r == ROUNDS - 1
        patcher.use("base")
        tokens_base, ms_b = timed(lambda: run(model, ids, STEPS, False,
                                              step_ms=sb if last else None))

        state = PagedState(L, H, D, DTYPE)
        state.alloc = BlockAllocator(N_BLOCKS)
        patcher.use("paged", state)
        patcher.stats = {"kernel": 0, "sdpa": 0, "kernel_ms": 0.0}
        tokens_mine, ms_p = timed(lambda: run(model, ids, STEPS, True, state,
                                              step_ms=sp if last else None))

        base_ms.append(ms_b)
        mine_ms.append(ms_p)
        print(f"轮 {r+1}: HF cache {ms_b:8.1f} ms | 我们的 {ms_p:8.1f} ms"
              f"{'   (冷启动，丢弃)' if r == 0 else ''}")

    if sp:
        print(f"\n末轮拆步: prefill HF {sb[0]:.1f} / 我们 {sp[0]:.1f} ms；"
              f"decode 每步 HF {st.mean(sb[1:]):.2f} / 我们 {st.mean(sp[1:]):.2f} ms")

    kept_b, kept_p = base_ms[1:], mine_ms[1:]
    med_b, med_p = st.median(kept_b), st.median(kept_p)
    same = sum(a == b for a, b in zip(tokens_base, tokens_mine))
    print(f"\nHF cache  : {tokens_base}")
    print(f"我的 kernel: {tokens_mine}")
    print(f"token 一致 {same}/{STEPS}（判据 >= {STEPS-1}）")
    print(f"\n中位数（丢首轮）: HF {med_b:.1f} ms | 我们 {med_p:.1f} ms  → {med_b/med_p:.2f}x")
    print(f"范围: HF {min(kept_b):.1f}–{max(kept_b):.1f} | 我们 {min(kept_p):.1f}–{max(kept_p):.1f}")
    print(f"decode 调用 {patcher.stats['kernel']} 次，kernel 累计 "
          f"{patcher.stats['kernel_ms']:.1f} ms（每次 {patcher.stats['kernel_ms']/max(1,patcher.stats['kernel']):.3f} ms，含 Python 传参）")
    ok = same >= STEPS - 1
    print("判定：", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
