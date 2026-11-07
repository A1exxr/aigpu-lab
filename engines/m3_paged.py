"""M3a：分页 KV 池 + block_table + 分配器（验证分页语义）

这一步**不换 attention**，还是 HF 算。目的只有一个：
    把 HF 的 KV cache 照抄进自己的分页池，再按 block_table 取回来，逐位比对。
    ── 先证明“块怎么分配、块内偏移、跨块拼接”是对的，再谈换 kernel。

判据：gather 回来的 k/v 与 HF cache 里的 k/v **逐位相同**（fp32 时 torch.equal 为 True）。

跑：
    HF_HUB_OFFLINE=1 ~/miniconda3/envs/triton/bin/python engines/m3_paged.py

填完先跑 fp32，全 True 之后再改 DTYPE = torch.float16 跑一遍
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")        # 模型已在本地缓存，不再联网
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "gpt2"
DTYPE = torch.float32          # 先 fp32 验语义；通过后改 torch.float16 再验一次
BLOCK_SIZE = 16                # 一页装几个 token（= kernel 里的 BS）
N_BLOCKS = 64                  # 池子里一共几页（够 64*16=1024 个 token）
STEPS = 8
PROMPT = "The history of the Roman Empire is long and complicated. "


# ---------------------------------------------------------------- 分配器
class BlockAllocator:
    """最简单的 free list：够用就行，不做碎片整理"""

    def __init__(self, n_blocks):
        self.n_blocks = n_blocks
        self.free = list(range(n_blocks))
        self.used = set()

    def allocate(self, n):
        # 从 self.free 里取 n 个块号，从 free 移除、加入 used、返回 list[int]
        # 不够就 raise RuntimeError（不要静默返回少几个）
        if n > len(self.free):
            raise RuntimeError(f'需要{n}块，只剩{len(self.free)}块')
        ids = self.free[:n]                 # 取前 n 个
        del self.free[:n]                   # 从 free 里移除
        self.used.update(ids)               # 加到 used
        return ids

    def free_blocks(self, ids):
        # 把 ids 还回 free（去重、排序随意，但要保持一致），从 used 移除
        for b in ids:
            if b in self.used:
                self.used.remove(b)
                self.free.append(b)

    @property
    def n_used(self):
        return len(self.used)


# ---------------------------------------------------------------- 池
class KVPool:
    """布局与 kernel 完全一致：每层两块 [N_BLOCKS, BLOCK_SIZE, H, D]"""
    def __init__(self, n_layers, n_heads, head_dim, dtype=DTYPE):
        self.H, self.D, self.BS = n_heads, head_dim, BLOCK_SIZE
        shape = (N_BLOCKS, BLOCK_SIZE, n_heads, head_dim)
        self.k = [torch.zeros(shape, dtype=dtype, device="cuda") for _ in range(n_layers)]
        self.v = [torch.zeros(shape, dtype=dtype, device="cuda") for _ in range(n_layers)]

    def write_token(self, layer, block_id, offset, k_tok, v_tok):
        """把 一个 token 的 k/v 写进池：k_tok/v_tok 形状 [H, D]。"""
        # 写到 self.k[layer][block_id, offset]，v 同理
        # 注意：offset 是块内偏移（0..BLOCK_SIZE-1），不是全局位置
        self.k[layer][block_id, offset] = k_tok
        self.v[layer][block_id, offset] = v_tok
        

    def gather(self, layer, block_table, length):
        """按 block_table 把前 length 个 token 的 k/v 拼回 [H, length, D]（注意维序）"""
        # 按 token 序号 t=0..length-1：
        # blk = block_table[t // BLOCK_SIZE]，off = t % BLOCK_SIZE
        # 取 self.k[layer][blk, off] → 形状 [H, D]，最后 stack 成 [H, length, D]
        # 提示：先收集成 list，再 torch.stack(dim=1)
        ks, vs = [], []
        for t in range(length):
            blk = block_table[t // self.BS]
            off = t % self.BS
            ks.append(self.k[layer][blk, off])
            vs.append(self.v[layer][blk, off])
        k = torch.stack(ks, dim=1)
        v = torch.stack(vs, dim=1)
        return k, v

    def nbytes(self):
        return sum(x.numel() * x.element_size() for x in self.k + self.v)


# ---------------------------------------------------------------- HF cache 读取
def layer_kv(cache, i):
    """不同 transformers 版本 cache 结构不一样，这里做兼容，取第 i 层的 (k, v)。"""
    if hasattr(cache, "key_cache") and len(cache.key_cache):          # 4.x / 5.x 旧式
        return cache.key_cache[i], cache.value_cache[i]
    if hasattr(cache, "layers"):                                      # 5.x 新式
        lay = cache.layers[i]
        return lay.keys, lay.values
    raise RuntimeError(f"不认识的 cache 类型：{type(cache)}，请把它的属性打印出来")


def main():
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE).eval().cuda()
    H = model.config.n_head
    D = model.config.n_embd // H
    L = model.config.n_layer
    print(f"gpt2: L={L} H={H} D={D} dtype={DTYPE}  池容量={N_BLOCKS*BLOCK_SIZE} token")

    pool = KVPool(L, H, D)
    alloc = BlockAllocator(N_BLOCKS)
    table = []                      # 本请求的 block_table
    ids = tok(PROMPT, return_tensors="pt").input_ids.cuda()
    past = None

    with torch.inference_mode():
        n_written = 0
        for step in range(STEPS):
            inp = ids if step == 0 else ids[:, -1:]
            out = model(inp, past_key_values=past, use_cache=True)
            past = out.past_key_values

            # ---- 1) 为这一步新产生的 token 分配块（跨块时多分一块）
            n_new = inp.shape[1]
            done = len(table) * BLOCK_SIZE          # 已写入的 token 数
            need = math.ceil((n_written + n_new) / BLOCK_SIZE) - len(table)
            if need > 0:
                table += alloc.allocate(need)

            # ---- 2) 把这一步的 k/v 照抄进池
            for layer in range(L):
                k, v = layer_kv(past, layer)        # [B, H, S_all, D]
                for j in range(n_new):
                    t = n_written + j                     # 该 token 的全局序号
                    blk, off = table[t // BLOCK_SIZE], t % BLOCK_SIZE
                    pool.write_token(layer, blk, off,
                                     k[0, :, t], v[0, :, t])

            # ---- 3) 从池里取回来，与 HF cache 逐位比
            for layer in range(L):
                k_hf, v_hf = layer_kv(past, layer)
                k_pool, v_pool = pool.gather(layer, table, n_written + n_new)
                if not torch.equal(k_pool, k_hf[0]):
                    print(f"[FAIL] L{layer} K 不一致，max={ (k_pool-k_hf[0]).abs().max().item():.3e}")
                    return
                if not torch.equal(v_pool, v_hf[0]):
                    print(f"[FAIL] L{layer} V 不一致")
                    return
            
            nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
            ids = torch.cat([ids, nxt], dim=1)
            n_written += n_new
            print(f"step {step}: 写入 {len(table)} 块 / {n_written} token，"
                  f"{L} 层 K/V 全部逐位一致 ✅")

    print(f"\n全部通过：{STEPS} 步 × {L} 层 K/V 与 HF 逐位相同")
    print(f"池显存 {pool.nbytes()/2**20:.1f} MiB，已用 {alloc.n_used} 块"
          f"（{alloc.n_used*BLOCK_SIZE} token 容量，实际 {ids.shape[1]} token）")
    print(f"碎片：分配了 {alloc.n_used} 块，最后一块只用 "
          f"{ids.shape[1] % BLOCK_SIZE or BLOCK_SIZE}/{BLOCK_SIZE} 个位置")


if __name__ == "__main__":
    main()
