"""M2 的统计版：交替测量 + 丢首次 + 2 倍剔除 + 中位数。

和 m2_kvcache.py 的区别只在"怎么测"，模型和循环逻辑完全一样。
用法：
    HF_HUB_OFFLINE=1 ~/miniconda3/envs/triton/bin/python engines/m2_kvcache_bench.py
"""

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")        # 模型已在本地缓存，不再联网
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import statistics as st

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "gpt2"
MAX_NEW = 20
ROUNDS = 5                          # 交替测 5 轮
PROMPT = "The history of the Roman Empire is long and complicated. " * 40


def load():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    model.eval().cuda()
    return tok, model


def timed(fn):
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    out = fn()
    e.record()
    torch.cuda.synchronize()
    return out, s.elapsed_time(e)


@torch.inference_mode()
def run_baseline(model, input_ids, max_new, step_ms):
    ids, tokens = input_ids, []
    for _ in range(max_new):
        logits, ms = timed(lambda: model(ids).logits)
        step_ms.append(ms)
        nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tokens.append(nxt)
        ids = torch.cat([ids, nxt], dim=1)
    return torch.cat(tokens, dim=1)


@torch.inference_mode()
def run_incremental(model, input_ids, max_new, step_ms):
    ids, tokens, past = input_ids, [], None
    for step in range(max_new):
        inp = ids if step == 0 else ids[:, -1:]
        out, ms = timed(lambda: model(inp, past_key_values=past, use_cache=True))
        step_ms.append(ms)
        past = out.past_key_values
        nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tokens.append(nxt)
        ids = torch.cat([ids, nxt], dim=1)
    return torch.cat(tokens, dim=1)


def robust(vals, label):
    """丢首次 → 按中位数 2 倍剔除 → 返回中位数与范围。"""
    kept = list(vals[1:])                                  # 丢首次（冷启动）
    med0 = st.median(kept)
    kept2 = [v for v in kept if v <= 2 * med0]
    dropped = len(kept) - len(kept2)
    med = st.median(kept2)
    print(f"{label:12s} 中位 {med:8.2f}  范围 {min(kept2):7.2f}–{max(kept2):7.2f}"
          f"  （{len(kept2)} 个样本，剔除 {dropped} 个 >2×中位）")
    return med


def main():
    torch.manual_seed(0)
    tok, model = load()
    input_ids = tok(PROMPT, return_tensors="pt").input_ids.cuda()
    S = input_ids.shape[1]
    print(f"prompt S = {S}；每轮 重算{MAX_NEW}步 → 增量{MAX_NEW}步，共 {ROUNDS} 轮交替")

    with torch.inference_mode():                           # 预热
        model(input_ids)
    torch.cuda.synchronize()

    base_tot, inc_tot, base_last, inc_last = [], [], [], []
    ref = None
    for r in range(ROUNDS):
        b_ms = []
        b = run_baseline(model, input_ids.clone(), MAX_NEW, b_ms)
        base_tot.append(sum(b_ms))
        base_last.append(b_ms[-1])

        i_ms = []
        i = run_incremental(model, input_ids.clone(), MAX_NEW, i_ms)
        inc_tot.append(sum(i_ms))
        inc_last.append(i_ms[-1])

        assert torch.equal(b, i), f"第 {r+1} 轮重算版与增量版不一致"
        if ref is None:
            ref = b
        else:
            assert torch.equal(ref, b), "不同轮之间输出不一致"

    print(f"\n逐位一致: True（{ROUNDS} 轮全部相同）")
    print("\n--- 20 步总耗时（ms）---")
    mb = robust(base_tot, "重算")
    mi = robust(inc_tot, "增量")
    print(f"总耗时加速比: {mb / mi:.2f}x")

    print("\n--- 稳态单步（末步，ms）---")
    lb = robust(base_last, "重算")
    li = robust(inc_last, "增量")
    print(f"稳态加速比: {lb / li:.2f}x")

    print("\n--- 绝对值（第 1 轮原始数）---")
    print(f"重算首步 {base_tot[0]:.1f} / 增量首步 {inc_tot[0]:.1f} ms")
    print("（首步含 prefill，两条路径都要吃完整 prompt，省不掉）")
    print(f"\n峰值显存: {torch.cuda.max_memory_allocated() / 2**20:.1f} MB")


if __name__ == "__main__":
    main()
