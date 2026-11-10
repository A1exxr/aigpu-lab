from __future__ import annotations              # 这代码必须在所有代码qian'm
"""M4：把引擎的 TTFT / TPOT / 吞吐 三个指数量出来。

复用 M3b-2 的引擎（`m3_paged_kernel.py`），只加计时与报表，不改任何计算。

三个指标的口径（写死，别改）
    TTFT  = 请求发出 → 第一个 token 可用（含 prefill）        = step_ms[0]
    TPOT  = 之后每个 token 的平均间隔                        = mean(step_ms[1:])
    吞吐  = tokens / 秒（**单请求**，含 prefill）             = 1000 * 步数 / sum(step_ms)

为什么要跑两档 prompt
    S=13 时 prefill 几乎不花时间，TTFT ≈ TPOT，看不出东西；
    必须再跑 S=441 那一档，prefill 的影响才显出来，也才给 M5 的 batching 留下尺子。

跑法
    python engines/m4_metrics.py            # 默认两档都跑
    python engines/m4_metrics.py --mult 40  # 只跑 S=441 那档
"""

'''
    优化前数据：
        档	        实现	        TTFT	    TPOT	    吞吐	            结论
        S=12	    HF cache	    5.22 ms	    4.70 ms	    219 tok/s	        —
        S=12	    我们	        8.59	    5.31	    176	                慢：TTFT +3.4ms，TPOT +0.6ms/步
        S=441	    HF cache	    6.94 ms	    4.31 ms	    227 tok/s	        —
        S=441	    我们	        130.63	    5.50	    83.7	            慢：TTFT +124ms（灾难），TPOT +1.2ms/步
    
    TTFT=130.63ms
    慢的原因：prefill 的 gather 在 Python 循环里做了 10584 次 select（12 层 × 441 token × 2 次 K/V），
            加上 sdpa 和 append 的开销，总共 130 ms。HF 的 prefill 一次 sdpa 就搞定 ———— prefill 的 k/v 本来连续，不需要 gather
            prefill 的 k/v 本来就是连续的，但我的实现强迫它“写进分页池再 gather 回来” ———— 多了一次不必要的往返

            同时 append 也是 12 层 × 441 token × 2 次，比 gather 还大
        append 441次 python 循环
        def append(self, layer, k_new, v_new):
            s_new = k_new.shape[1]                 # 441（prefill）

            for j in range(s_new):                 # 循环 441 次
                t = start + j                      # 全局 token 序号
                blk = self.table[t // BLOCK_SIZE]  # 逻辑页 → 物理块
                off = t % BLOCK_SIZE               # 页内偏移
                self.pool.write_token(layer, blk, off, k_new[:, j], v_new[:, j])
        每次 write_token里：
            self.k[layer][blk, off] = k_tok    # 一次 __setitem__
            self.v[layer][blk, off] = v_tok    # 又一次 __setitem__
        共882次__setitem__，12层10584次，每一次都是 Python 解释器 → PyTorch dispatch → 边界检查 → CUDA kernel 启动 → 写入 → 返回——~7~10 μs，大部分是 CPU 开销
    改：
        池子可以被看成  1024 行 × [H, D]  的二维表
        self.k[layer]    # [N_BLOCKS, BLOCK_SIZE, H, D] = [64, 16, 12, 64] 
        前两维是 块号 * 页内偏移 ————  64 块 × 16 位置 = 1024 个槽，把前两维压扁成一维[1024, 12, 64]
        通过一次kernel并行操作，把441个token写到各自的441行，而不是441次python调用
        
        
        prefill 不走 gather，直接用c_attn输出的连续k/v算spda，同时写进池
'''



import argparse
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")        # 模型已在本地缓存，不再联网
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import statistics as st
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from m3_paged import N_BLOCKS, BlockAllocator  # noqa: E402
from m3_paged_attn import PagedState  # noqa: E402
from m3_paged_kernel import (  # noqa: E402
    DTYPE,
    MODEL,
    STEPS,
    AttnPatcher,
    build_kernel,
    run,
    timed,
)

ROUNDS = 3            # 交替测量轮数，第 1 轮丢弃
MULTIPLES = (1, 40)   # prompt 重复倍数：1 → S=13，40 → S=441

# 与 m3_paged_kernel.PROMPT 同一句；不复用它是因为它会被环境变量 PMUL 预先乘过，
# 而这里要靠 --mult 自己控制长度。
PROMPT_BASE = "The history of the Roman Empire is long and complicated. "


def prompt_ids(tok, mult: int):
    return tok(PROMPT_BASE * mult, return_tensors="pt").input_ids.cuda()


def measure(model, patcher, ids, paged: bool):
    """交替跑 ROUNDS 轮，返回 (末轮逐 step 毫秒, 各轮总毫秒, 末轮 token)

    交替（而不是先跑完 A 再跑完 B）是为了让两边的温度/频率漂移互相抵消
    """
    step_ms, totals, tokens = [], [], None
    for r in range(ROUNDS):
        last = r == ROUNDS - 1
        buf = [] if last else None

        if paged:
            cfg = model.config
            L = cfg.n_layer
            H = cfg.n_head
            D = cfg.n_embd // cfg.n_head

            state = PagedState(L, H, D, DTYPE)
            state.alloc = BlockAllocator(N_BLOCKS)
            patcher.use("paged", state)
            patcher.stats = {"kernel" : 0, "sdpa" : 0, "kernel_ms" : 0.0}
            
            tokens, ms = timed(lambda: run(model, ids, STEPS, True, state, step_ms=buf))
        else:
            patcher.use("base")
            tokens, ms = timed(lambda: run(model, ids, STEPS, False, step_ms=buf))

        totals.append(ms)
        if last:
            step_ms = buf
        print(f"    轮 {r + 1}: {ms:8.1f} ms{'   (冷启动，丢弃)' if r == 0 else ''}")
    return step_ms, totals, tokens


def metrics(step_ms, total_ms):
    """由逐 step 毫秒算出三个指标，返回 (ttft, tpot, tput)"""
    
    ttft = step_ms[0]                                # 第一步耗时（含 prefill）
    tpot = st.mean(step_ms[1:])                      # decode 每步平均
    tput = 1000.0 * len(step_ms) / total_ms          # tokens/秒（单请求，含 prefill）
    return ttft, tpot, tput


def report(rows):
    """打印一张 6 行的表：2 档 prompt × 2 个实现。"""
    print("\n" + "=" * 78)
    print(f"{'prompt':>8} {'实现':>8} {'TTFT(ms)':>10} {'TPOT(ms)':>10} "
          f"{'吞吐(tok/s)':>12} {'token 一致':>10}")
    print("-" * 78)

    # ✅ 补上：打印每一行
    for r in rows:
        print(f"{r['mult']:>8} {r['impl']:>8} {r['ttft']:>10.2f} {r['tpot']:>10.2f} "
              f"{r['tput']:>12.1f} {r['same']:>10}")

    print("=" * 78)

     # 对每档 mult 检查 base 与 paged 的 token 是否一致
    by_mult = {}
    for r in rows:
        by_mult.setdefault(r["mult"], {})[r["impl"]] = r

    print("\n判定：")
    for mult, pair in by_mult.items():
        base_same = pair["HF cache"]["same"]
        paged_same = pair["我们的"]["same"]
        n_steps = len(pair["HF cache"]["steps"])
        ok = base_same >= n_steps - 1
        print(f"  S={mult}: token 一致 {base_same}/{n_steps}  "
              f"{'PASS' if ok else 'FAIL'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mult", type=int, nargs="*", default=list(MULTIPLES),
                    help="prompt 重复倍数，1 → S=13，40 → S=441")
    args = ap.parse_args()

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE).eval().cuda()
    cfg = model.config
    print(f"gpt2: L={cfg.n_layer} H={cfg.n_head} D={cfg.n_embd // cfg.n_head} "
          f"dtype={DTYPE}，编译 kernel ...")
    mod = build_kernel()
    patcher = AttnPatcher(model, mod)

    rows = []
    for mult in args.mult:
        ids = prompt_ids(tok, mult)
        print(f"\n[prompt 重复 {mult} 次 → S={ids.shape[1]}，生成 {STEPS} 步，"
              f"{ROUNDS} 轮交替]")
        per_impl = {}
        for impl, paged in (("HF cache", False), ("我们的", True)):
            step_ms, totals, tokens = measure(model, patcher, ids, paged)
            kept = totals[1:]                      # 丢第 1 轮
            total_ms = st.median(kept)
            ttft, tpot, tput = metrics(step_ms, total_ms)
            per_impl[impl] = tokens
            rows.append(dict(mult=ids.shape[1], impl=impl, ttft=ttft, tpot=tpot,
                             tput=tput, steps=step_ms,
                             same=None, total=total_ms,
                             rng=(min(kept), max(kept))))
        n_same = sum(a == b for a, b in zip(per_impl["HF cache"], per_impl["我们的"]))
        for r in rows[-2:]:
            r["same"] = n_same

    report(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
