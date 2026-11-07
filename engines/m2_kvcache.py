"""台阶 2（M2）：手写增量解码 —— 只把"新 token"喂进模型，历史靠 KV cache。

M1 的写法每步都把整段序列重新 forward 一遍（累计 O(S²)）。
M2 改成：第 1 步吃整段 prompt，之后每步只吃 1 个 token，历史 K/V 从 cache 里取。

判据（两条，缺一不可）：
    ① 20 个 token 与 M1 / HF 逐位一致
    ② 每步耗时基本恒定（不随 S 增长）—— 在 S≈1024 时才看得出来，S<32 全是启动开销

跑法：
    HF_HUB_OFFLINE=1 ~/miniconda3/envs/triton/bin/python engines/m2_kvcache.py
"""

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")        # 模型已在本地缓存，不再联网
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "gpt2"
MAX_NEW = 20
PROMPT = "The history of the Roman Empire is long and complicated. " * 40   # 凑到 ~500 token
SYMBOLS = "B batch / S 序列长度 / D head 维度 / H head 数 / L 层数 / V 词表"


def load():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    model.eval().cuda()
    return tok, model


@torch.inference_mode()
def run_baseline(model, input_ids, max_new):
    """M1：每步重算整段，用来当对照 """
    ids = input_ids
    tokens, step_ms = [], []
    for _ in range(max_new):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        logits = model(ids).logits
        end.record()
        torch.cuda.synchronize()
        step_ms.append(start.elapsed_time(end))

        nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tokens.append(nxt)
        ids = torch.cat([ids, nxt], dim=1)
    return torch.cat(tokens, dim=1), step_ms


@torch.inference_mode()
def run_incremental(model, input_ids, max_new):
    """M2：历史交给 KV cache，每步只喂 1 个 token

    model(...) 支持 use_cache=True，返回一个 past_key_values 对象；
    下一步把它原样传回去，并且 input_ids 只喂"最新那 1 个 token"。
    position 会由 cache 自动推进，不需要你手动算。
    """
    tokens, step_ms = [], []
    past = None
    ids = input_ids

    for step in range(max_new):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        # 调用 model(..., use_cache=True)，第 1 步喂整段 ids，
        # 之后只喂 ids[:, -1:]；把 past 传进去、再收回来。
        if past is None:
            # 第 1 步 prefill，整段prompt
            out = model(ids, use_cache=True)
        else:
            # 后续 decode，每次只喂最新的1个token和历史cache
            # ids[:,-1:] 是切片，结果是 [B, 1] ———— 取最后一列，保持二维，形状 [B, 1] ； model 需要二维输入 [B, S]
            # ids[:, -1] 是整数索引，取最后一列，消掉第 1 维	[B]（一维）
            out = model(ids[:,-1:], past_key_values = past, use_cache=True)

        end.record()
        torch.cuda.synchronize()
        step_ms.append(start.elapsed_time(end))

        # 从 out 里取 logits 的最后一个位置，argmax 得到下一个 token
        nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        # 把 nxt 拼到 ids 后面；同时把 out.past_key_values 记到 past 里
        ids = torch.cat([ids, nxt], dim=1)      # ids保留完整历史
        tokens.append(nxt)
        past = out.past_key_values              # 更新 cache

    return torch.cat(tokens, dim=1), step_ms


def kv_cache_bytes(S, B=1, H=12, D=64, L=12, dtype_bytes=4):
    """KV cache 的显存账：2（K 和 V）× L 层 × B × H × S × D × 字节数"""
    return 2 * L * B * H * S * D * dtype_bytes


def main():
    torch.manual_seed(0)
    tok, model = load()
    input_ids = tok(PROMPT, return_tensors="pt").input_ids.cuda()
    prompt_len = input_ids.shape[1]
    print(f"prompt 长度 S = {prompt_len}")
    print(f"符号：{SYMBOLS}")
    print(f"gpt2：L=12 H=12 D=64 V=50257；KV cache 每 token = "
          f"{kv_cache_bytes(1) / 1024:.0f} KiB（fp32）")

    with torch.inference_mode():                      # 预热
        model(input_ids)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    base, base_ms = run_baseline(model, input_ids, MAX_NEW)
    base_total = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    inc, inc_ms = run_incremental(model, input_ids, MAX_NEW)
    inc_total = (time.perf_counter() - t0) * 1000

    print("\n--- 正确性 ---")
    print("重算版:", repr(tok.decode(base[0])))
    print("增量版:", repr(tok.decode(inc[0])))
    print("逐位一致:", torch.equal(base, inc))

    print("\n--- 单步耗时（ms）---")
    print("重算版  首/末:", f"{base_ms[0]:.2f} / {base_ms[-1]:.2f}")
    print("增量版  首/末:", f"{inc_ms[0]:.2f} / {inc_ms[-1]:.2f}")
    print(f"总耗时  重算 {base_total:.1f} ms  →  增量 {inc_total:.1f} ms")
    print(f"加速比: {base_total / inc_total:.2f}x")
    print(f"峰值显存: {torch.cuda.max_memory_allocated() / 2**20:.1f} MB")
    print(f"KV cache（S={prompt_len + MAX_NEW}）应占: "
          f"{kv_cache_bytes(prompt_len + MAX_NEW) / 2**20:.1f} MB")


if __name__ == "__main__":
    main()
