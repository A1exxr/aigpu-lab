"""台阶 1（M1）：手写 greedy 解码循环，与 HF generate 逐位对账。

符号表（全项目统一，与 kernels/ 里 CUDA 代码一致，不再换名字）：
    B   batch 大小                      （本例 1）
    S   序列长度 = 已生成的 token 数      （kernel 里叫 S，HF 文档里叫 T，我们用 S）
    D   head 维度                       （gpt2: 768/12 = 64）
    H   head 数                         （gpt2: 12）
    L   层数                            （gpt2: 12）
    V   词表大小                         （gpt2: 50257）
形状约定：
    input_ids / ids : [B, S]            （整数 token id）
    logits          : [B, S, V]         （每个位置对全词表的打分）
    KV cache        : [B, H, S, D]      × L 层 × 2（K 和 V）

纯 PyTorch，不写 CUDA、不做 KV cache —— 目的是先把"一条能对账的基线"立起来。
判据：20 个 token 与 HF 完全一致。

跑法（必须先跑 _tools/_setup_m1.sh 装好依赖）：
    cd ~/projects/aigpu-lab
    ~/miniconda3/envs/triton/bin/python engines/m1_naive.py
"""

import os
import time

# 首次运行要从 HF 下 500 MB 模型，国内建议先设镜像：
#   export HF_ENDPOINT=https://hf-mirror.com
# 已经下过、想跳过启动时的联网检查：
#   HF_HUB_OFFLINE=1 ./m1_naive.py
# 但不要把离线开关写进代码——别人 clone 后没有缓存，会直接报错。

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "gpt2"
MAX_NEW = 20
PROMPT = "The capital of France is"


def load():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    model.eval().cuda()
    return tok, model


@torch.inference_mode()
def my_generate(model, input_ids, max_new):
    """最笨的写法：每步都把整段序列重新 forward 一遍。"""
    ids = input_ids
    new_tokens = []
    records = []
    for step in range(max_new):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        logits = model(ids).logits                 # [B, S, V]
        end.record()
        torch.cuda.synchronize()            # record 是异步的，要等GPU跑完
        
        ms = start.elapsed_time(end)        # 毫秒

        nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)   # greedy：取最大

        new_tokens.append(nxt)
        ids = torch.cat([ids, nxt], dim=1)

        records.append((step, ms, ids.shape[1],
                       torch.cuda.max_memory_allocated() / 2**20))       # 峰值显存（MB）

        
    return torch.cat(new_tokens, dim=1), records


def hf_generate(model, input_ids, max_new):
    return model.generate(
        input_ids,
        max_new_tokens=max_new,
        do_sample=False,                              # greedy
        pad_token_id=model.config.eos_token_id,
    )


def main():
    torch.manual_seed(0)
    tok, model = load()
    input_ids = tok(PROMPT, return_tensors="pt").input_ids.cuda()
    prompt_len = input_ids.shape[1]

    # 预热：第一次 forward 要初始化 cuBLAS 句柄 / 加载 kernel，不能算进单步时间
    with torch.inference_mode():
        model(input_ids)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    mine, records = my_generate(model, input_ids, MAX_NEW)
    t_mine = (time.perf_counter() - t0) * 1000

    ref_full = hf_generate(model, input_ids, MAX_NEW)
    ref = ref_full[:, prompt_len:]

    print("prompt :", repr(PROMPT))
    print("我的   :", repr(tok.decode(mine[0])))
    print("HF     :", repr(tok.decode(ref[0])))
    print("逐位一致:", torch.equal(mine, ref))
    print(f"手写循环总耗时: {t_mine:.1f} ms")
    print("峰值显存: %.1f MB" % (torch.cuda.max_memory_allocated() / 2**20))

    print("\n步  S      ms    峰值显存")
    for step, ms, S, mem in (records[1], records[-1]):   # 跳过第 1 步（仍含冷启动）
        print(f"{step + 1:2d}  {S:2d}  {ms:7.2f}  {mem:6.1f} MB")
    print(f"首末倍数: {records[-1][1] / records[1][1]:.2f}x")
    print("答：实测每步 7~9 ms、几乎不随 S 变化 —— 这个规模下（gpt2、S<32）耗时被"
          "kernel 启动与框架开销主导，O(S²) 的重算还没显形；S 上百才会看出来。"
          "这也说明：小模型上想靠 KV cache 省时间，省的不是算力而是重算的份额。")


if __name__ == "__main__":
    main()
