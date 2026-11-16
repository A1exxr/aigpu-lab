/*
 * w12_fa2.cu —— FlashAttention-2 前向（causal + GQA）
 *
 * 目标：和 F.scaled_dot_product_attention(is_causal=True) 对上（fp16，atol 2e-3）
 *
 * 与 w5_flash_attn.py（FA1 思路）的差别，改的就是这四处：
 *   1. Q 常驻寄存器，内层循环不再每轮把 Q 从 smem 读一遍
 *   2. warp 沿 Q 的行切（一个 warp 负责 16 行），不是沿 D 切
 *   3. 因果：整块被 mask 的 K/V tile 直接跳过（不是算完再丢）
 *   4. 输出只在最后除一次 l（acc 里累积未归一化的值）
 *
 * 线程划分（128 线程，4 个 warp）：
 *   warp = tid / 32        -> 负责第 warp*16 .. warp*16+15 行
 *   lane = tid % 32
 *     lane & 15  -> 行内第几行（0..15）
 *     lane >> 4  -> 这个线程负责 D 的哪一半（0 -> d[0..31]，1 -> d[32..63]）
 *   所以同一行的两个线程是 lane 和 lane^16，用 __shfl_xor_sync(..., 16) 合并
 *
 * 每个线程的寄存器存放：
 *      Q 的当前行	    float q[32] 
 *      输出累积        float acc[32]
 *      softmax状态     float m 和 l
 * Q 和 acc 在寄存器里，这是 FA2 的核心优化
 */

#include <cuda_fp16.h>

#define TILE_M 64          // 每个 block 处理多少行 Q
#define TILE_N 32          // 每次读多少列 K/V
#define HEAD_DIM 64          // head_dim
#define SM_THREADS 128

#define NEG_INF (-1e30f)

// 强制内联
// 让下面类型转换 写法简便点 ———— h2f(x) 和 __half2float(x)；  这里的 __half 就是 half
__device__ __forceinline__ float h2f(__half h) { return __half2float(h); }

/*
    __launch_bounds_(NTHREADS)是CUDA给编译器的一个 编译期提示
    告诉编译器：这个 kernel 启动时，每个线程块最多就是 SM_THREADS 个线程
                编译器据此来决定每个线程可以分配多少寄存器，在“寄存器够用”和“occupancy 别太低”之间做权衡
    
    核心作用：约束寄存器分配
        如果编译器不知道你的块有多大，它会假设最坏情况（通常是 1024 线程/块），从而非常保守地分配寄存器，
        导致每个线程拿到的寄存器很少，或者反过来，如果不加约束，编译器可能为了减少指令数而给每个线程分配大量寄存器，
        导致 SM 上能同时驻留的线程块变少，occupancy 下降，延迟隐藏能力变差
*/ 
__global__ void __launch_bounds__(SM_THREADS) fa2_forward_kernel(const __half* __restrict__ Q,
                                                                const __half* __restrict__ K,
                                                                const __half* __restrict__ V,
                                                                __half* __restrict__ O,
                                                                int S, int H, int Hkv, float scale)
{
    /*
        Attention 里面有Q/K/V头的配比方式，按K/V头从多到少排列，核心区别就是：多少个query头共享一个K/V头
        1、标准 MHA (Multi-Head Attention)
            q 头:   q0   q1   q2   q3
                    │    │    │    │
            kv 头:  k0   k1   k2   k3
                    v0   v1   v2   v3

            H = 4, Hkv = 4, 比例 1:1
                每个 q 头都有自己独立的k和v头，一一对应
                表达力最强，但 K/V 的参数量和显存最大 ———— K/V cache 在推理时要存 H 份
        
        2、MQA（Multi-Query Attention）
            所有 q 头共享同一个k头和v头；K/V cache 缩小 H 倍，推理时显存和带宽大降；表达力下降，效果可能略差

        3、GQA（Grouped-Query Attention）
            MHA 和 MQA 的折中：q 头分组，每组共享一个 kv 头
            q 头:   q0   q1  │  q2   q3
                    │    │   │   │    │
                    └────┘   │   └────┘
                       │     │      │
            kv 头:     k0    │     k1
                       v0    │     v1

            H = 4, Hkv = 2, 比例 2:1
                H 个 q 头 ———— Hkv 个 kv 头 ———— H 是 Hkv 的整数倍
                每 grp = H / Hkv 个连续 q 头共享一个 kv 头
                K/V cache 缩小 H / Hkv 倍，同时保留比 MQA 更多的表达力

            GQA 是 Google 2023 年提出的，Llama 2/3、Mistral、Qwen 等主流大模型都用它。它基本取代了 MQA，成为“省 K/V 又不掉点”的标准方案

            这里从 x 到 Q/K/V 进行映射的时候，W_k / W_v 的输出维度被砍了
            W_q: [d_model, H   * d_head]   → Q: [B, H,   S, d_head]
            W_k: [d_model, Hkv * d_head]   → K: [B, Hkv, S, d_head]
            W_v: [d_model, Hkv * d_head]   → V: [B, Hkv, S, d_head]
            
            d_model = 4096, H = 32, Hkv = 8时，K/V 的参数量和 cache 都缩小到1/4，省下来的就是推理时的 K/V cache 和带宽
            少了的kv还是可以和32个q头进行配对，grp = H / Hkv = 32 / 8 = 4，每 4 个连续 q 头共享一个 kv 头
            h = 0,1,2,3 → kvh = 0，都读 K[b, 0]；
            h = 4,5,6,7 → kvh = 1，都读 K[b, 1]；
            ... 每个q头仍然可以做完整的[S,D] @ [D,S]，只是4个q头读了同一份K，并不是靠广播和维度拓展
    */
    // S=512, B=2, H=8, Hkv=2, D=64 
    // K/V: [B, Hkv, S, D]
    // 对每个[b,h] 算 O[b,h] = softmax(Q[b,h] @ K[b,kvh]ᵀ * scale + causal_mask) @ V[b,kvh]
    // 1个block的全部任务：读 Q 的 64 行 → 循环读 K/V 的所有块 → online softmax 累积 → 写回 O 的 64 行
    // GQA映射 和 基址
    const int bh  = blockIdx.y;              // 0 .. B*H-1
    const int b   = bh / H;                  // 第几个 batch
    const int h   = bh % H;                  // 这个 batch 里的 第几个头(q头)
    const int grp = H / Hkv;                 // 几个 q 头共用一个 kv 头 ———— 每组有几个 q 头
    const int kvh = h / grp;                 // kv头的索引 ———— 当前 q 头 h 对应的 kv 头索引，范围 [0, Hkv)

    // 这里 Q [B,H,S,D] 一共有2*8=16个[S,D]矩阵，也就是 16 个 [512,64] 矩阵
    // gird(8, 2*8)  一共 128个block
    // blockIdx.y 选哪个矩阵，blockIdx.x 选这个矩阵的哪 64 行
    // 128 个 block 各自处理 16 个不同 (batch, q 头) 矩阵的 64 行 
    const int row_base = blockIdx.x * TILE_M;                           // 本 block 的第一行，一个block处理 TILE_M 行的 q

    const size_t q_base = ((size_t)b * H   + h  ) * S * HEAD_DIM;      // Q/O 的基址
    const size_t k_base = ((size_t)b * Hkv + kvh) * S * HEAD_DIM;       // K/V 的基址

    // 线程划分
    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    // 1个block处理64行的q，1个block有128线程，为4个warp，则一个warp处理16行的q
    // 2个lane负责q的一行
    const int row_local = warp_id * 16 + (lane & 15);       // 本线程负责的行（block 内） 0 1 2 ... 63

    // lane 0 和 lane 16 是同一行（行 0）的两个线程，一个管前半 D，一个管后半 D
    // 这里拆开是因为只有128线程，负责 64 行 Q；如果一个线程管一整行 64 维，warp 利用率直接砍半，而且每个线程要存 float q[64];float acc[64];还有其他变量，会溢出
    const int d_base = (lane >> 4) * 32;                    // 0或32 ———— 本线程负责的 D 的哪一半，D的起始偏移
    const int row  = row_base + row_local;

    __shared__ __half Ks[TILE_N][HEAD_DIM];
    __shared__ __half Vs[TILE_N][HEAD_DIM];

    /*
        1 个 block 生命周期：
            ① 把 Q 的 64 行装进寄存器（整段只读一次 Q）
            ↓
            ② 循环 K/V tile（按 TILE_N=32 列一块）：
                2.1 把 K/V tile 搬进 shared memory
                2.2 算 score s = q·k * scale（拆 D + shuffle 合并）
                2.3 causal mask（未来位置置 -inf）
                2.4 online softmax 更新 m/l/acc
                    ↓
            ③ 写回：acc / l，一次性写到 O

        关键设计：Q 常驻寄存器，不进 smem、不每轮重读；K/V 每轮搬 smem，供 block 内所有线程共享
    */
    // ---- 1. Q 装进寄存器（整段唯一一次读 Q）----
    // 每个线程读自己那 32 维：d_base + d，d_base 是 0 或 32
    float q[32];
    #pragma unroll
    for (int d = 0; d < 32; ++d) 
    {
        // row < S 最后一块不满时，越界行读0
        q[d] = (row < S) ? h2f(Q[q_base + (size_t)row * HEAD_DIM + d_base + d]) : 0.f;
    }

    // 初始化3个状态 acc  m  l
    /*
        写成 float acc[32] = {0.f}; 是可以的
        这里写成循环展开是因为，数组可能会被放进 local memory
        如果编译器不能证明“这个数组的每个元素都能被独立地放进寄存器”，它就会把整个数组放到 local memory
        手写 for + #pragma unroll
        循环被完全展开后，编译器看到的是 32 个独立的标量赋值 acc[0]=0; acc[1]=0; ...，每个元素都是独立变量，更容易全部提升到寄存器
    */ 
    float acc[32];
    #pragma unroll
    for (int d = 0; d < 32; ++d) acc[d] = 0.f;
    float m = NEG_INF;   // running max
    float l = 0.f;       // running sum of exp

    // ---- 2. 因果：只需要走到对角块为止 ----
    // 对应 triton 的 hi = min(start_m + Block_m, S)；block最后一行的全局行号
    const int row_last = min(row_base + TILE_M - 1, S - 1);     

    /*
        TILE_N = 32 是K块的列块，全局列号 j 落在哪个tile里，编号就是 j / TILE_N 向下取整
        j ∈ [0, 32)   → j/32 = 0   → tile 0
        j ∈ [32, 64)  → j/32 = 1   → tile 1
        j ∈ [64, 96)  → j/32 = 2   → tile 2

        row_last 是本 Q 块最后一行能看到的 Score矩阵的最后一列，它落在哪个 tile，那个 tile 就是 jt_end
        比如：
            row_last = min(64+63, 199) = 130
            jt_end   = 130 / 32 = 4          （整数除法，向下取整）
            jt = 0, 1, 2, 3，4                   → 扫 tile 0,1,2,3,4
            覆盖列 = [0, 5*32) = [0, 160)
            但本Q块只看到了130列，后面的列 131..159 属于未来，扫了它们确实是“算了未来”
            但这些格子会被 元素级mask 置为NEG_INF，softmax权重为0，不会污染结果
            
            这就是 块级截断 和 元素级mask 的配合
            块级只能粗到 tile，tile 内部分无效必须靠元素 mask 兜住
    */
    const int jt_end   = row_last / TILE_N;     // 覆盖 row_last 的K块编号，循环 jt <= jt_end
    
    // jt 表示 j tile，第 j 个 K/V 列 tile
    for (int jt = 0; jt <= jt_end; ++jt) 
    {
        // 2.1 把本轮的 K/V tile 搬进共享内存
        // 共 TILE_N * HEAD_DIM = 2048个元素，128 线程各搬 16 个
        for (int idx = threadIdx.x; idx < TILE_N * HEAD_DIM; idx += SM_THREADS) 
        {
            const int jj = idx / HEAD_DIM;      // 元素在K/V tile 里的的行索引      
            const int dd = idx % HEAD_DIM;      // 列索引
            const int j  = jt * TILE_N + jj;    // K/V 的全局行索引
            const bool ok = (j < S);
            Ks[jj][dd] = ok ? K[k_base + (size_t)j * HEAD_DIM + dd] : __float2half(0.f);
            Vs[jj][dd] = ok ? V[k_base + (size_t)j * HEAD_DIM + dd] : __float2half(0.f);
        }
        __syncthreads();

        // 2.2 算这一块的分数 s[jj] = q·k[jj] * scale，再用 XOR 16 合并另一半
        float s[TILE_N];        // 线程负责的那一行query对第32行key的分数
        #pragma unroll
        for (int jj = 0; jj < TILE_N; ++jj) 
        {
            float partial = 0.f;    // 本线程这 32 维的内积
            #pragma unroll
            for(int d = 0; d < 32; ++d)
            {
                // 这份K被64行Q反复读
                partial += q[d] * h2f(Ks[jj][d_base + d]);
            }

            // 合并 同一行的另一半（lane ^ 16）
            // __shfl_xor_sync() 交换寄存器值
            // 参1：0xffffffffu，32个lane全部参与
            // xor模式，每个lane发出自己的 partial 值，并收到 lane+16 的值
            const float other = __shfl_xor_sync(0xffffffffu, partial, 16);
            // 合并后 线程就可以拿到完整的score去算自己的acc，
            // 比如 lane0 和 lane16，算自己的acc[32],lane0算的是前32维，lane16是后32维
            // 写回后才是完整的64维
            s[jj] = (partial + other) * scale;

            // 因果 mask 
            // 被遮住的位置要变成 NEG_INF，让 exp() 变 0
            int j = jt * TILE_N + jj;
            if(j > row || j >= S)       // 元素级mask
            {
                s[jj] = NEG_INF;
            }
        }

        // 2.3 online softmax
        // m_new = max(m, max_jj s[jj])
        // alpha = exp(m - m_new)
        // l     = l * alpha + sum_jj exp(s[jj] - m_new)
        // acc[d] = acc[d] * alpha + sum_jj exp(s[jj] - m_new) * V[jj][d0+d]
        // 注意：acc 的缩放必须在累加 PV 之前；p = exp(s-m_new) 只算一次，别算两遍
        float m_new = m;
        #pragma unroll
        for(int jj = 0; jj < TILE_N; ++jj)
        {
            m_new = fmaxf(m_new, s[jj]);
        }
        float alpha = expf(m - m_new);

        float p[TILE_N];
        float l_new = l * alpha;
        #pragma unroll
        for(int jj = 0; jj < TILE_N; ++jj)
        {
            p[jj] = expf(s[jj] - m_new);
            l_new += p[jj];
        }
        l = l_new;

        // 这里acc[32]是跟着D维走的，32=HEAD_DIM/2,所以循环变量用d表示
        #pragma unroll
        for(int d = 0; d < 32; ++d)
        {
            acc[d] *= alpha;
        }
        
        #pragma unroll
        for(int jj = 0; jj < TILE_N; ++jj)
        {
            #pragma unroll
            for(int d = 0; d < 32; ++d)
            {
                acc[d] += p[jj] * h2f(Vs[jj][d_base + d]);
            }
        }

        m = m_new;
        __syncthreads();
    }

    // ---- 3. 写回：只在最后除一次 l ----
    if(row < S)
    {
        #pragma unroll
        for(int d = 0; d < 32; ++d)
        {
            O[q_base + (size_t)row * HEAD_DIM + d_base + d] = __float2half(acc[d] / l);
        }
    }
}



#include <torch/extension.h>

torch::Tensor fa2_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, double scale)
{
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "q/k/v must be CUDA");
    TORCH_CHECK(q.dtype() == torch::kHalf, "q must be fp16");
    TORCH_CHECK(q.dim() == 4, "q must be [B,H,S,D]");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(), "need contiguous");

    const int B = q.size(0), H = q.size(1), S = q.size(2), D = q.size(3);
    const int Hkv = k.size(1);
    TORCH_CHECK(D == HEAD_DIM, "head_dim must be ", HEAD_DIM);
    TORCH_CHECK(H % Hkv == 0, "H must be a multiple of Hkv");

    auto o = torch::empty_like(q);

    // 每个 block 负责：某个 (b, h) 的、Q 的某 64 行，算出这 64 行的输出
    dim3 grid((S + TILE_M - 1) / TILE_M, B * H);
    fa2_forward_kernel<<<grid, SM_THREADS>>>(
        reinterpret_cast<const __half*>(q.data_ptr()),
        reinterpret_cast<const __half*>(k.data_ptr()),
        reinterpret_cast<const __half*>(v.data_ptr()),
        reinterpret_cast<__half*>(o.data_ptr()),
        S, H, Hkv, (float)scale);
    return o;
}
