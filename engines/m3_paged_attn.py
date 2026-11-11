"""M3b-1：把 HF 的 attention 换掉，K/V 从我们自己的分页池读（仍是 PyTorch 算）

M3a 证明了 写进池 → 按 block_table 取回来 是逐位对的

这一步再往前一步：不再让 HF 管 cache，每个 token 的 k/v 写进池，
attention 的 QKᵀ / softmax / PV 由我们自己算，喂给下游的还是同形状的 [B,S,768]

判据（fp32，零容差）：
    8 步 greedy 的 token 序列与 M2（HF 原生 cache） 完全一致；
    logits 的 max|diff| 记下来（应该是 1e-5 量级，来自 matmul 结合顺序不同）

跑：
    HF_HUB_OFFLINE=1 ~/miniconda3/envs/triton/bin/python engines/m3_paged_attn.py

这一步通了，M3b-2 只做一件事：把这里的
`torch.matmul` 换成我们的 `paged_attn_split` kernel（fp16）
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")        # 模型已在本地缓存，不再联网
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import math

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from m3_paged import BlockAllocator, KVPool, BLOCK_SIZE, N_BLOCKS

MODEL = "gpt2"
DTYPE = torch.float32
STEPS = 8
PROMPT = "The history of the Roman Empire is long and complicated. "


class PagedState:
    """一次请求的分页状态：池 + 分配器 + block_table + 已写入的 token 数"""

    def __init__(self, n_layers, n_heads, head_dim, dtype, pool=None, alloc=None):
        # 当前设计有缺陷：当前代码是“每个请求一个独立池”，不是“全局共享池”，因为每个 PagedState 都会新建一个池和分配器
        # 多请求会实例化多份池，各占 75 MB，互不共享
        # vllm是 池 和 分配器 全局共享，每请求一张 block_table
        # M5：池/分配器可以由外部注入（多请求共享一份），不传时退回「每请求一个池」
        self.pool = pool if pool is not None else KVPool(n_layers, n_heads, head_dim, dtype)
        self.alloc = alloc if alloc is not None else BlockAllocator(N_BLOCKS)
        self.table = []                                             # 本请求用到的块号，按顺序
        self.n = 0                                                  # 已写入池的 token 数

        # M4 优化参数
        self._idx_new = None        # 本步新 token 的行号 [s_new]，12 层复用；write_run 用
        self._idx_all = None        # 全步 token 的行号 [n]，decode 的 gather_run 用

    def append(self, layer, k_new, v_new):
        """k_new/v_new: [H, s_new, D]，把新增的 token 追加进池"""
        # prefill（step 0）时 s_new = prompt_len，decode 时 s_new = 1
        s_new = k_new.shape[1]      # 这一步新产生几个 token
        
        # larye 0 时，self.n是旧值(这一步之前的总数)，这里要区分 层layer 和 步step
        # layer 0 是每步forward里面的第0层(12层模型的第1层)
        # step 0 是整个生成过程的第0步，也就是prefill
        # layer 1+ 时，self.n已经被外部推进去，是新值
        start = self.n if layer == 0 else self.n - s_new    
        # 这里self.n在layer 0 append之后就变成 state.n += S了，所以layer 1+时，要减s_new

        # self.n 是所有层共享的计数，只能在 layer==0 时推进
        # 只有 layer 0 负责分配块，表结构只跟 token 位置有关，跟层无关
        if layer == 0:
            need = math.ceil((self.n + s_new) / BLOCK_SIZE) - len(self.table)
            if need > 0:
                self.table += self.alloc.allocate(need)         # += 就是 table.extend()

        # # 把 s_new 个 token 逐个写进池
        # for j in range(s_new):
        #     t = start + j                                       # 整个序列的第几号 token
        #     blk = self.table[t // BLOCK_SIZE]
        #     off = t % BLOCK_SIZE
        #     # 一样的省略了:   k_new[:, j, :]  
        #     # 取这一步的所有新token的K/V，s_new 个 token，每个都是[H, D]
        #     self.pool.write_token(layer, blk, off, k_new[:, j], v_new[:, j])

        # M4优化写法：
        # 单 token 还是走 write_token
        if s_new == 1:      
            t = start
            self.pool.write_token(layer,
                                  self.table[t // BLOCK_SIZE], t % BLOCK_SIZE,
                                  k_new[:, 0], v_new[:, 0]) 

        # 批量走 write_run
        else:
            if layer == 0:
                # 索引全局共享，只在 layer 0 算一次 _idx_new 和 _idx_all
                # tab: Python list ——> GPU 张量
                tab = torch.tensor(self.table, dtype=torch.int64, device = self.pool.k[0].device)     # 块数

                # 本步 token 的全局序号  [0, 1, ..., 440]
                t_new = torch.arange(start, start + s_new, device=tab.device)        
                # 本步 token 的物理行号  形状也是[441]，但值是具体的物理行号，不是逻辑序号
                self._idx_new = tab[t_new // BLOCK_SIZE] * BLOCK_SIZE + (t_new % BLOCK_SIZE)

                # 全部 token 的全局序号  
                t_all = torch.arange(0, start + s_new, device = tab.device)   
                # 全部 token 的物理行号          
                self._idx_all = tab[t_all // BLOCK_SIZE] * BLOCK_SIZE + (t_all % BLOCK_SIZE)

            # 12 层都走到这里，复用上面算好的索引
            self.pool.write_run(layer,
                                self._idx_new,
                                # 这里要transpose是write_run 要求 source 的 dim0 是 token
                                # [H, s_new, D] → [s_new, H, D]
                                k_new.transpose(0,1),
                                v_new.transpose(0,1))
            

    def gather(self, layer):
        # 如果_idx_all是None，或长度不对，就重建，否则就用缓存的_idx_all
        if self._idx_all is None or self._idx_all.numel() != self.n:
            tab = torch.tensor(self.table, dtype=torch.int64, device=self.pool.k[0].device)
            t_all = torch.arange(0, self.n, device=tab.device)
            self._idx_all = tab[t_all // BLOCK_SIZE] * BLOCK_SIZE + (t_all % BLOCK_SIZE)
        return self.pool.gather_run(layer, self._idx_all)


def patch_attention(model, state):
    """把每一层的 attn.forward 换掉：k/v 走我们的池，attention 我们自己算

    关键：必须用「工厂函数」把 layer_idx / attn / H / D / scaling 绑成参数
    直接在 for 循环里定义 patched 并用循环变量，是 Python 的经典坑——
    闭包捕获的是**变量名**不是值，循环结束后所有 patched 看到的 layer_idx 都是 11
    """

    blocks = model.transformer.h if hasattr(model, "transformer") else model.h

    # 工厂函数，返回另一个函数或对象的 函数
    # 生成一个 patched函数 并返回
    def make_patched(layer_idx, attn, H, D, scaling):
        # patched 内部，每层 attention 的实际计算
        # patched 的签名，必须和HF原生 attn.forward 一致，不然传的参数对不上
        def patched(hidden_states, past_key_values=None, attention_mask=None,
                    output_attentions=False, **kw):         # **kw是接受任何多余的关键字参数，但不使用
            B, S, _ = hidden_states.shape           # hidden_states[B, S, 768]

            # 1) 用 HF 自己的投影算 q/k/v（权重不动，只换 cache 与 attention 的算法）
            qkv = attn.c_attn(hidden_states)            # [B, S, 3E]  E=768        这里不用写第几层的qkv投影，参数 attn 里面有12层各自的qkv投影，layer_idx：写哪层的池；attn用哪层权重；两者必须成对
            q, k, v = qkv.split(attn.split_size, dim=2)          # 沿着第二维拆开，每个都是 [B, S, E]，这里的split_size是HF里面自己定义好的，值就是768
            # E = H * D = 768   H = 12  D = 64，768维被拆成 12个头*64维
            # view就是把768按照H=12，D=64拆开
            q = q.view(B, S, H, D).transpose(1, 2)          # [B,S,H,D] ——> [B,H,S,D]
            k = k.view(B, S, H, D).transpose(1, 2)
            v = v.view(B, S, H, D).transpose(1, 2)

            # 2) k/v 进池（layer 0 时推进全局计数）
            # k[0]：去掉 batch 维，[B,H,S,D] → [H,S,D] ———— append 的签名要 [H, s_new, D]
            state.append(layer_idx, k[0], v[0])     
            if layer_idx == 0:
                state.n += S                        # 先推进n，再gather，此时n是当前总数
            '''
                知识点： 关于这里的 state 
                    patched 定义时：完全不知道 state 是什么 ———— 只是“引用一个叫 state 的变量”
                    patched 只关心 state 有没有 append/gather 方法，不关心它是不是 PagedState 类型
                    patched 被调用时：执行到 state.append(...)，这时才去 state 指向的对象上找 append 方法
                    如果 main 里没实例化就传进来，运行时调 state.append(...) 会直接 AttributeError

                    因为 python 是动态型语言，在运行时才会检查变量类型；C++是静态类型，写代码时就会报错
            '''

            # 3) 从池里取回全部 K/V：[B,H,n,D]
            K, V = state.gather(layer_idx)          # [H,n,D]，取回的是包含这一步新token的完整KV
            K, V = K.unsqueeze(0), V.unsqueeze(0)   # 在dim=0插入新维度 [H, n, D] ——> [1, H, n, D]， F.sdpa 要求 [B, H, n, D]

            # 4) attention 用 PyTorch 内置的 attention SDPA算子
            out = F.scaled_dot_product_attention(q, K, V, is_causal=(S > 1), scale=scaling)   # [B,H,S,D]       
            out = out.transpose(1, 2).reshape(B, S, H * D)      # [B,H,S,D] ——> [B,S,H,D] ——>  [B,S,768]
            out = attn.c_proj(out)      # attn.c_proj：HF 的输出投影（Linear(768,768)） 用相同的权重和偏置
            return out, None

        return patched

    # 12层循环，每层调用一次工厂函数
    # 调用工厂，返回“绑定了这一层 layer_idx, attn, H, D, scaling 的 patched
    for layer_idx, block in enumerate(blocks):
        attn = block.attn           # attn 是 GPT2Attention的实例
        # make_patched返回一个函数对象，让 block.attn.forward 走我的 patched 方法
        block.attn.forward = make_patched(layer_idx, attn,
                                          attn.num_heads, attn.head_dim, attn.scaling)


def main():
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE).eval().cuda()
    cfg = model.config          # model.config是模型结构配置对象，装着所有超参数(层数、头数、hidden size、词表大小...)
    H, D, L = cfg.n_head, cfg.n_embd // cfg.n_head, cfg.n_layer
    print(f"gpt2: L={L} H={H} D={D} dtype={DTYPE}")

    ids = tok(PROMPT, return_tensors="pt").input_ids.cuda()

    # ---- 基线：M2 的做法（HF 原生 DynamicCache）----
    with torch.inference_mode():
        base = ids.clone()      # 深拷贝
        past = None
        for step in range(STEPS):
            inp = base if step == 0 else base[:, -1:]
            o = model(inp, past_key_values=past, use_cache=True)
            past = o.past_key_values
            base = torch.cat([base, o.logits[:, -1, :].argmax(-1, keepdim=True)], dim=1)
    # base是ids的形状：[B, S]，第0维取0，第1维取不包括prompt的token ———— batch 0 的 8 个新 token
    base_tokens = base[0, ids.shape[1]:].tolist()

    # ---- 对照：我们的分页池 + 自算 attention ----
    state = PagedState(L, H, D, DTYPE)
    patch_attention(model, state)
    with torch.inference_mode():
        cur = ids.clone()
        for step in range(STEPS):
            inp = cur if step == 0 else cur[:, -1:]
            # position_ids 必须自己给：我们传 past_key_values=None，
            # HF 会认为 没有历史，然后每步位置都从 0 开始，decoder 看到的位置全错，所以要手动设置 真实位置pos(这一步每个token在序列的绝对位置)
            start = state.n
            # pos是 [1, S] 的整数张量，这一步的 S 个 token，各自在全局序列里排第几号
            pos = torch.arange(start, start + inp.shape[1], device=cur.device).unsqueeze(0)   # 补 batch 维度
            o = model(inp, past_key_values=None, use_cache=False,
                      position_ids=pos)                          # 不再用 HF cache
            cur = torch.cat([cur, o.logits[:, -1, :].argmax(-1, keepdim=True)], dim=1)
    my_tokens = cur[0, ids.shape[1]:].tolist()

    print(f"\nHF cache  : {base_tokens}")
    print(f"我的池    : {my_tokens}")
    ok = base_tokens == my_tokens
    print(f"token 完全一致: {ok}")
    print(f"池显存 {state.pool.nbytes()/2**20:.1f} MiB，{len(state.table)} 块装 {state.n} token")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
