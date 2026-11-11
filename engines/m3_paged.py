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
N_BLOCKS = 64                  # 池子里一共64个物理块（够 64*16=1024 个 token）
STEPS = 8
PROMPT = "The history of the Roman Empire is long and complicated. "


# ---------------------------------------------------------------- 分配器
# 分配器 管 哪些块是空闲的
# BlockAllocator 只管“块号”——不碰任何 KV 数据，只维护 free 和 used 两个集合
class BlockAllocator:
    """最简单的 free list：够用就行，不做碎片整理"""

    def __init__(self, n_blocks):
        self.n_blocks = n_blocks
        # list保持顺序，allocate 取前n个 ———— list 切片 [:n] 是 O(n)，而且保持分配顺序（先分配的块号小）
        self.free = list(range(n_blocks))       # 空闲块号，按顺序排
        self.used = set()                       # 已用块号，O(1) 查，判断这块在不在used

    def allocate(self, n):
        # 从 self.free 里取 n 个块号，从 free 移除、加入 used、返回 list[int]
        # 不够就 raise RuntimeError（不要静默返回少几个）
        if n > len(self.free):
            raise RuntimeError(f'需要{n}块，只剩{len(self.free)}块')
        ids = self.free[:n]                 # 取前 n 个
        # del是原地修改，所有引用这个 list 的地方都看到变化
        del self.free[:n]                   # 从 free 里移除
        # update()是批量添加，接受一个可迭代对象，把里面每个元素都加进去；add() 只加一个元素
        self.used.update(ids)               # 加到 used，存的是具体的块号
        return ids

    def free_blocks(self, ids):
        # 把 ids 还回 free（去重、排序随意，但要保持一致），从 used 移除
        for b in ids:
            if b in self.used:      # 防止重复释放
                self.used.remove(b)
                self.free.append(b)

    @property
    def n_used(self):
        return len(self.used)       # 返回used了的个数


# ---------------------------------------------------------------- 池
# 池 管 KV存哪里
# KVPool 只管“KV 张量”——不关心谁在用哪块，只负责“给我块号+偏移，我读写”
# KVPool(...) 分配 75 MB（64 块 × 16 位置 × 12 层 × 12 头 × 64 维 × 2 × 4B）
class KVPool:
    """布局与 kernel 完全一致：每层两块 [N_BLOCKS, BLOCK_SIZE, H, D]"""
    def __init__(self, n_layers, n_heads, head_dim, dtype=DTYPE, n_blocks=None):
        self.H, self.D, self.BS = n_heads, head_dim, BLOCK_SIZE
        # M5：块数可由调用方给（多请求共享一个池）；不传时退回 M3/M4 默认的 64
        self.n_blocks = N_BLOCKS if n_blocks is None else n_blocks

        # (64个物理块，每块装16个token，12个头，每个头64维)
        shape = (self.n_blocks, BLOCK_SIZE, n_heads, head_dim)

        # gpt2有12层，每层有自己的 attention，每层的 K/V 是独立的
        # attention 是逐层算的，第0层的 attention 只需要第0层的 K/V
        # 所以池里要存 12层 * 2(K和V)个分页张量
        # 分开存，kernel 里每层只读自己的那一个张量
        # self.k是一个列表，里面就是12个张量 ———— 例如：torch.zeros([64, 16, 12, 64]) 第 0 层的 K
        # 这里把整个池子按照层数，分成了12个张量
        self.k = [torch.zeros(shape, dtype=dtype, device="cuda") for _ in range(n_layers)]      # 等价于 for + append
        self.v = [torch.zeros(shape, dtype=dtype, device="cuda") for _ in range(n_layers)]

    def write_token(self, layer, block_id, offset, k_tok, v_tok):
        """把 一个 token 的 k/v 写进池：k_tok/v_tok 形状 [H, D]"""
        # k_tok 和 v_tok 是从 HF cache 里取出的一个token的k/v
        # 写到 self.k[layer][block_id, offset]，v 同理
        # offset 是块内偏移（0..BLOCK_SIZE-1）
        # 比如t=20，block_table = [3, 7]，   block_id = 20//16=1; offset = 20 % 16 = 4
        # t=20放到页1(物理块7，这里假设刚开始的prompt，table就给了3和7两个物理块，不一定是连续的)，第4个位置

        # self.k[layer] 形状：[N_BLOCKS, BLOCK_SIZE, H, D] = [64, 16, 12, 64]
        # self.k[layer][blk, off]    # 索引前两维 → [H, D] = [12, 64]
        
        self.k[layer][block_id, offset] = k_tok
        self.v[layer][block_id, offset] = v_tok
        

    def gather(self, layer, block_table, length):
        """按 block_table 把分散在物理块里前 length 个 token 的k/v 收集回一个连续张量 [H, length, D]（注意维序）
            但收集成逻辑连续的张量，是一次全量拷贝，n越大，拷贝越贵
            每层 attention 之前都调，attention 需要“逻辑连续”的 K/V ———— q @ Kᵀ 要求 K 的序列维按 token 顺序排好
        """
        # length 为 当前实际写入了多少个 token
        # 按 token 序号 t=0..length-1：
        # blk = block_table[t // BLOCK_SIZE]，off = t % BLOCK_SIZE
        # 取 self.k[layer][blk, off] → 形状 [H, D]，最后 stack 成 [H, length, D]
        # 提示：先收集成 list，再 torch.stack(dim=1)
        ks, vs = [], []
        for t in range(length):
            blk = block_table[t // self.BS]         # 逻辑页号 ——> 物理块号
            off = t % self.BS                       # 页内偏移
            ks.append(self.k[layer][blk, off])
            vs.append(self.v[layer][blk, off])
        # 循环结束，ks是 length 个 [H, D] 的tensor
        # stack(dim=1)新建一个维度，插在第一维，形状为[H, length, D]
        # result[h, t, d] = ks[t][h, d]，语义本身没变：第 t 个输入 tensor 的 [h, d] 元素，原封不动地成为结果的 [h, t, d] 元素
        # 逻辑坐标的映射是确定的，可预测的，对应关系没变
        # ks[0][0,0]（值 1）永远是 result[0,0,0]    ks[1][0,0]（值 5）永远是 result[0,1,0]
        # 因为要和 HF cache 的 [H, S, D] 维度顺序对齐
        k = torch.stack(ks, dim=1)          # gather返回的k v是没有batch维度的
        v = torch.stack(vs, dim=1)
        return k, v


    '''新增两个方法，优化M4
        write_run 和 gather_run
    '''
    def write_run(self, layer, idx, k, v):     
        '''
            prefill 一批 token 一次性写进池

        idx: [len] int64 —— 每个 token 在「摊平后的一维坐标」上的行号
             行号 = 物理块号 * BLOCK_SIZE + 块内偏移

        k/v: [len, H, D] —— 本步新算出来的 K/V（已按 token 排在前面）

        '''
        # scatter：一批 token 一次写进池
        # self.k[layer] 形状为[N_BLOCKS, BLOCK_SIZE, H, D] = [64, 16, 12, 64]
        # 通过view方法把前两维 合并成 行 ——> [64*16, 12, 64] = [1024, 12, 64]
        # index_copy_0(dim=0,idx,k)  ———— scatter写入
        # 语义：target[idx[i]]=source[i]   这里source就是k
        # idx[i] 是第 i 个 token 的物理行号，k[i] 是第 i 个 token 的 [H, D] ———— index_copy_ 沿着第 0 维 把 k[i] 写到第 idx[i] 行
        # 池子一共有1024行可以写，但M4 的 prefill 最多S=441，所以写不满
        self.k[layer].view(-1, self.H, self.D).index_copy_(0, idx, k)  
        self.v[layer].view(-1, self.H, self.D).index_copy_(0, idx, v)  
        '''
            比如：这里k的每个元素都是[12,64]
            idx = [48, 49, 112, 116]（4 个 token），k 是 [4, 12, 64...]
            k[0] → 写到 target 第 48 行
            k[1] → 写到 target 第 49 行
            k[2] → 写到 target 第 112 行
            k[3] → 写到 target 第 116 行
            一次 kernel 完成 4 次写入——不是 4 次 __setitem__
        '''
        
    def gather_run(self, layer, idx):
        # 把池子看成1024行 * [H,D]
        kf = self.k[layer].view(-1, self.H, self.D)                     # [1024, 12, 64]
        vf = self.v[layer].view(-1, self.H, self.D)                     # [1024, 12, 64]

        # index_select(dim=0,dix)
        # 语义：result[i]=target[idx[i]]，和index_copy_相反
        # index_select 返回 [len, H, D]（每行一个 token)
        # transpose(0, 1) 换到 [H, len, D] ———— 和 HF 的 [B, H, S, D] 去掉 batch 后一致
        # transpose返回的是 view，不连续的
        k = kf.index_select(0, idx).transpose(0, 1).contiguous()
        v = vf.index_select(0, idx).transpose(0, 1).contiguous()
        return k, v     #  [H, len, D]



    def nbytes(self):
        return sum(x.numel() * x.element_size() for x in self.k + self.v)


'''
        k = tensor([[
            # batch 0
            [[1, 2, 3, 4],          # h=0, t=0
            [5, 6, 7, 8],           # h=0, t=1  ← 要这个
            [9,10,11,12]],          # h=0, t=2

            [[13,14,15,16],         # h=1, t=0
            [17,18,19,20],          # h=1, t=1  ← 要这个
            [21,22,23,24]]          # h=1, t=2
        ]])
        # 形状 [1, 2, 3, 4] = [B, H, S, D]

        k[0, :, 1]
        # = [[5, 6, 7, 8],          # h=0 的 t=1
        #    [17,18,19,20]]         # h=1 的 t=1
        # 形状 [2, 4] = [H, D] 
'''


# ---------------------------------------------------------------- HF cache 读取
def layer_kv(cache, i):
    """不同 transformers 版本 cache 结构不一样，这里做兼容，取第 i 层的 (k, v)"""
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
    H = model.config.n_head                 # 头数 12
    D = model.config.n_embd // H            # 768/12 = 64
    L = model.config.n_layer                # 层数 12
    print(f"gpt2: L={L} H={H} D={D} dtype={DTYPE}  池容量={N_BLOCKS*BLOCK_SIZE} token")     # 池容量：64*16=1024token；是整个池的容量，不是每个请求的
    # 多个请求共享一个池，每个请求维护自己的页表

    pool = KVPool(L, H, D)                  # 创建分页池
    alloc = BlockAllocator(N_BLOCKS)        # 创建分配器
    table = []                              # 本请求的 block_table；每份请求都有自己的映射表 table，但这里只实例化了一份
    ids = tok(PROMPT, return_tensors="pt").input_ids.cuda()
    past = None

    with torch.inference_mode():            # 关梯度，省显存
        n_written = 0                       # 到目前为止，实际写进池里的 token 总数(包括prompt)
        for step in range(STEPS):
            inp = ids if step == 0 else ids[:, -1:]     # 第 0 步时，喂整段 prompt，n_new就是20，n_written就是20了
            out = model(inp, past_key_values=past, use_cache=True)  # 第0步 forward 用到了past
            # out.past_key_values 永远返回"截至当前步的完整 KV
            past = out.past_key_values

            # ---- 1) 为这一步新产生的 token 分配块（跨块时多分一块）
            n_new = inp.shape[1]            # 这一步新产生几个 token
            # 分配用 n_written 算 需要几页
            # need 不是“这一步需要几页”，而是“总数需要几页 − 已经分配了几页”
            # 新 token 大概率落在旧块的空位里，所以和 n_written 有关
            need = math.ceil((n_written + n_new) / BLOCK_SIZE) - len(table)
            if need > 0:
                table += alloc.allocate(need)

            # ---- 2) 把这一步的 k/v 照抄进池
            # k_tok 和 v_tok 的 block_id 和 offset 是通过 全局 token 序号 t 映射出来的
            for layer in range(L):          # 同一个 token 在 12层里的t一样，算出的blk off也一样，但张量不一样，每一层都有一个k/v张量
                k, v = layer_kv(past, layer)        # [B, H, S_all, D]
                for j in range(n_new):
                    t = n_written + j                     # t是整个序列的第几个token，不是第几个新token，t是包含prompt的
                    blk = table[t // BLOCK_SIZE]
                    off = t % BLOCK_SIZE
                    # k[0]	            batch 0	                                        [H, S, D] = [12, S, 64]
                    # k[0, :, t]	    batch 0、第 t 个 token、所有头，所有维度	        [H, D] = [12, 64]
                    # k[0, :, t, :]	    同上（显式写 :）	                                [H, D]
                    pool.write_token(layer, blk, off, k[0, :, t], v[0, :, t])       

            # ---- 3) 从池里取回来，与 HF cache 逐位比
            for layer in range(L):
                k_hf, v_hf = layer_kv(past, layer)
                k_pool, v_pool = pool.gather(layer, table, n_written + n_new)   # 当前总 token 数
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
