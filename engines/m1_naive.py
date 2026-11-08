"""台阶 1（M1）：手写 greedy 解码循环，与 HF generate 逐位对账。

符号表（全项目统一，与 kernels/ 里 CUDA 代码一致，不再换名字）：
    B   batch 大小                      
    S   序列长度 = 已生成的 token 数      （kernel 里叫 S，HF 文档里叫 T）
    D   head 维度                       （gpt2: 768/12 = 64）
    H   head 数                         （gpt2: 12）
    L   层数                            （gpt2: 12）
    V   词表大小                         （gpt2: 50257）
形状约定：
    input_ids / ids : [B, S]            （整数 token id）
    logits          : [B, S, V]         （每个位置对全词表的打分）
    KV cache        : [B, H, S, D]  × L 层 × 2（K 和 V）

纯 PyTorch，不写 CUDA、不做 KV cache —— 目的是先把"一条能对账的基线"立起来。
判据：20 个 token 与 HF 完全一致。

跑法（必须先跑 _tools/_setup_m1.sh 装好依赖）：
    cd ~/projects/aigpu-lab
    ~/miniconda3/envs/triton/bin/python engines/m1_naive.py
"""
# 首次运行要从 HF 下 500 MB 模型，国内建议先设镜像：
#   export HF_ENDPOINT=https://hf-mirror.com
# 已经下过、想跳过启动时的联网检查：
#   HF_HUB_OFFLINE=1 ./m1_naive.py
# 但不要把离线开关写进代码——别人 clone 后没有缓存，会直接报错。

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")        # 模型已在本地缓存，不再联网
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer            # 加载模型 和 分词器 的包

MODEL = "gpt2"
MAX_NEW = 20                                    # 只生成20个token
PROMPT = "The capital of France is"             # prompt


def load():
    # tok：把文本切成 token id（"The capital" → [464, 3139]）
    tok = AutoTokenizer.from_pretrained(MODEL)                                          # 加载分词器，和MODEL用同一个名字，保证 token id 对齐
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)            # 加载模型，dtype是新版参数
    model.eval().cuda()             # eval()关掉dropout，不然greedy的结果可能和HF不一致；   cuda()权重搬到GPU，不搬的话 CPU/GPU 混算会报错或极慢
    return tok, model


'''
    out = model(ids)   out 是一个 CausalLMOutputWithCrossAttentions 类，里面有多个字段
    
    字段	                            形状	                含义
    out.logits	                    [B, S, V]	    每个位置对全词表的打分 ← 你要的
    out.hidden_states	tuple of    [B,S,768]	    每层的 hidden state（默认不返回）
    out.attentions	                  tuple	        每层的 attention 权重（默认不返回）
    out.past_key_values	              tuple	        KV cache（use_cache=True 时返回）

    logits = model(ids).logits
    logits 是  [B, S, V] = [1, S, 50257]
    logits[b, s, v]： 第b个样本、第s个位置，模型认为下一个token是词表第v个词的分数
    例如：
        假设 ids = [[1, 5]]
        logits[0, 0, :]  → 50257 个数，是"token 1 后面该跟什么"的打分
        logits[0, 1, :]  → 50257 个数，是"token 5 后面该跟什么"的打分  ← 只有这个有用

    假设词表只有5个词，logits[0, -1, :] 可能是：tensor([ 2.1, -1.3,  8.7,  0.5, -2.0])
    里面就是5个原始分数，logits 经过 softmax后才是概率
    logits 是 softmax 的输入， softmax 是 logits 的归一化
    
    记住：logits = 最后一层线性层（lm_head）的输出 = softmax 之前的原始分数。
    naive在每个前向都把所有token算一遍logits，但只需要最新的token的原始分数或数据，就是 看完整个序列后，下一个是什么
    所以只需要取[:,-1,:]，取最后一个位置的那一行的V个分数
    naive的lm_head 算了 S * 768 * 50527，但只用到了1 * 768 * 50527
'''



# @torch.inference_mode() 关梯度 + 更省的推理模式，推理首选
@torch.inference_mode()
def my_generate(model, input_ids, max_new):
    """最笨的写法：每步都把整段序列重新 forward 一遍"""
    ids = input_ids             # 当前已有的完整序列，一开始就就是prompt
    new_tokens = []             # 存放新生成的token，最后拼起来返回
    records = []                # 存放每步的性能属性

    # for循环跑20次，每次生成一个token
    for step in range(max_new):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        logits = model(ids).logits                                              # 前向，返回[B, S, V] 
        end.record()
        torch.cuda.synchronize()                                                # record 是异步的，要等GPU跑完
        
        ms = start.elapsed_time(end)                                            # 毫秒

        # [B, S, V] 里 只有最后一个位置有用，前面 S-1 个位置的 logits 全扔 ———— 所有的batch都取各自的最后一个位置(S-1)的token分数
        # argmax：返回最大值所在的索引；dim=-1，沿着最后一维找，保持维度不消
        # nxt是一个[1,1]的张量，就是[B,1] ———— 比如 tensor([[257]])
        nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)                     # greedy：取最大；keepdim=True 保持 [B,1]，方便 cat； 
        new_tokens.append(nxt)                                                  # 最后是20个[1,1]的张量组成的list

        # ids: [B, S]， 原来 S 个 ———— nxt: [B, 1]， 新 1 个  ————> 沿着 dim=1 维度拼接后 [B, S+1]
        ids = torch.cat([ids, nxt], dim=1)                                      # 拼到序列末尾；dim=1 是序列维，不是 batch 维

        records.append((step, ms, ids.shape[1],
                       torch.cuda.max_memory_allocated() / 2**20))              # 峰值显存（MB）= 全局峰值显存 / 2**20

        # torch.cat(new_tokens, dim=1) 是把 new_tokens 这个 list 里的所有 tensor 拼起来，不是"new_tokens 和某个东西拼"
        # new_tokens是一个list，列表只是一个容器，没有维度概念，所以dim=1，指的是容器里每个tensor的第一维
    return torch.cat(new_tokens, dim=1), records


# HF 的参考实现
def hf_generate(model, input_ids, max_new):
    return model.generate(
        input_ids,
        max_new_tokens=max_new,
        do_sample=False,                              # greedy
        pad_token_id=model.config.eos_token_id,
    )


def main():
    torch.manual_seed(0)                                                            # 固定随机种子
    tok, model = load()                                                             # 加载tokenizer 和 模型

    # 文本 ——> token id
    # return_tensors="pt"：返回 PyTorch tensor
    # .input_ids：取 token id 字段
    # .cuda()：搬到 GPU 和 model 同设备
    input_ids = tok(PROMPT, return_tensors="pt").input_ids.cuda()
    prompt_len = input_ids.shape[1]         # 记录 prompt 长度

    # 预热：第一次 forward 要初始化 cuBLAS 句柄 / 加载 kernel，不能算进单步时间
    # with：自动执行“进入”和“退出”
    with torch.inference_mode():        # 进入：关梯度 + 关 version counter
        model(input_ids)                
    # 退出：恢复原状
    torch.cuda.synchronize()            # 等预热跑完

    t0 = time.perf_counter()
    mine, records = my_generate(model, input_ids, MAX_NEW)
    t_mine = (time.perf_counter() - t0) * 1000

    # ref_full 形状: [1, prompt_len + 20] = [1, 25]   ← 含 prompt！
    ref_full = hf_generate(model, input_ids, MAX_NEW)
    # ref 形状: [1, 20]   ← 切掉前 prompt_len 个，只剩新生成的
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
