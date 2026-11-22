/*
    w12d_fa2_tile128.cu —— 在 w12c（8.79 TFLOPS）
    w12d 改成 8 个 warp + 每个 warp 只计算到自己的 causal 边界 + K/V 用 cp.async 双缓冲后台加载
    其他（swizzle、mma.sync 布局、P 打包、online softmax）完全沿用

*/

#include <cuda_fp16.h>

#define TILE_M     128              // Q [128,64]       一个block负责的Q，一个block 里有 8个warp
#define TILE_N     32               // K/V [32,64]
#define HEAD_DIM   64
#define SM_THREADS 256
#define WARP_M     16               // 每个 warp 还是负责 Q 的 16 行   m16n8k16
#define NEG_INF    (-1e30f) 
#define SMEM_BYTES (TILE_M * HEAD_DIM * 2 + 4 * TILE_N * HEAD_DIM * 2)   // Qs 16 KB + 2 组 (K,V) 双缓冲

// 负责 fp32 ——> fp16 打包成 32 位寄存器
__device__ __forceinline__ unsigned pack2(float a, float b)
{
    const __half2 h = __halves2half2(__float2half_rn(a), __float2half_rn(b));
    return *reinterpret_cast<const unsigned*>(&h);
}

// 4 条 lane 的 max 归约
__device__ __forceinline__ float warp_max4(float v)
{
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
    return v;
}

// 4 条 lane 的 sum 归约
__device__ __forceinline__ float warp_sum4(float v)
{
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    v += __shfl_xor_sync(0xffffffffu, v, 2);
    return v;
}

// 128B swizzle 重排，保证 8 个线程同时读不同 bank
__device__ __forceinline__ int swz(int r, int c) 
{
    return r * HEAD_DIM + ((((c >> 3) ^ (r & 7)) << 3) | (c & 7)); 
}


__global__ void __launch_bounds__(SM_THREADS) fa2_mma_kernel(const half* __restrict__ Q,
                                                            const half* __restrict__ K,
                                                            const half* __restrict__ V,
                                                            half* __restrict__ O,
                                                            int B, int H, int Hkv, int S, float scale)
{
    extern __shared__ char smem[];
    half* Qs = (half*)smem;                     // [TILE_M][HEAD_DIM]
    // K/V 双缓冲的统一基址
    half* KVb = Qs + TILE_M * HEAD_DIM;         // 双缓冲区：[buf][K/V][TILE_N][HEAD_DIM]
    // Kbuf[0] ———— KVb + 0 ———— K 矩阵 Buffer 0
    // Vbuf[0] ———— KVb + TILE_N × HEAD_DIM ———— V 矩阵 Buffer 0
    // Kbuf[1] ———— KVb + 2 × TILE_N × HEAD_DIM ———— K 矩阵 Buffer 1
    half* Kbuf[2] = {KVb,                     KVb + 2 * TILE_N * HEAD_DIM};
    half* Vbuf[2] = {KVb + TILE_N * HEAD_DIM, KVb + 3 * TILE_N * HEAD_DIM};        // Vbuf[1] 就是 V 矩阵 Buffer 1

    const int bh   = blockIdx.y;
    const int b    = bh / H;
    const int h    = bh - b * H;
    const int grp  = H / Hkv;
    const int kvh  = h / grp;

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;               // 现在1个block有8个warp
    const int lane    = tid & 31;
    const int g       = lane >> 2;              // 行号 0 ... 7
    const int tcol    = (lane & 3) * 2;         // 起始列 0 2 4 6
    const int gmask   = ((lane / 8) & 1) * 8;   // ldmatrix 的列组偏移    0 8 0 8 


    // 当 S < TILE_M 时，S - TILE_M 为负，max(0, ...) 将其截断到 0
    const int row_base = max(0, min(blockIdx.x * TILE_M, S - TILE_M));
    const int qrow     = row_base + warp_id * WARP_M;

    const size_t q_base = ((size_t)b * H   + h  ) * S * HEAD_DIM;
    const size_t k_base = ((size_t)b * Hkv + kvh) * S * HEAD_DIM;

    // ---- Q 进 smem ----
    for (int i = tid; i < TILE_M * HEAD_DIM; i += SM_THREADS)
    {
        const int r = i / HEAD_DIM;
        const int d = i - r * HEAD_DIM;
        Qs[swz(r, d)] = Q[q_base + (size_t)(row_base + r) * HEAD_DIM + d];
    }
    __syncthreads();

    // ---- Q ——> 4 个 A 片段，Q 在 KV循环中只加载一次，驻留寄存器----
    unsigned aq[HEAD_DIM / 16][4];
    for (int kk = 0; kk < HEAD_DIM / 16; ++kk)
    {
        // 行 = lane % 16        0..15，覆盖 16 行
        // 列组 = lane / 16      0 或 1，选左半（列 0..7）或右半（列 8..15）
        const int row_swz_q = warp_id * WARP_M + (lane % 16);
        const int col_swz_q = kk * 16 + (lane / 16) * 8;
        const half* p = Qs + swz(row_swz_q, col_swz_q);
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                     : "=r"(aq[kk][0]), "=r"(aq[kk][1]), "=r"(aq[kk][2]), "=r"(aq[kk][3])
                     : "r"((unsigned)__cvta_generic_to_shared(p)));
    }

    // ---- online softmax 状态 + O 累加器（全在寄存器）----
    float m0 = NEG_INF, m1 = NEG_INF, l0 = 0.f, l1 = 0.f;
    float acc[HEAD_DIM / 8][4];
    for (int nd = 0; nd < HEAD_DIM / 8; ++nd)
        for (int c = 0; c < 4; ++c) 
        {
            acc[nd][c] = 0.f;
        }

    const int row0  = qrow + g;
    const int row1  = qrow + g + 8;
    const int jt_end = min(row_base + TILE_M - 1, S - 1) / TILE_N;

    // ---- cp.async 双缓冲：16B 一块，正好对应一个 swizzle 的粒度 ----
    // 将当前 tile 对应的 K/V 数据从 HBM 异步搬入 SMEM 双缓冲区中的指定 buffer

    /*
        关于 双缓冲 和 cp.async
        双缓冲是“软件技巧”（两块 buffer 交替），cp.async 是“硬件指令”（异步拷贝）
        双缓冲不必用 cp.async，但不用 cp.async 的双缓冲基本是“假重叠”  ————  真正快的是“双缓冲 + cp.async”
        普通搬：
            load(jt+1, buf^1);    // LDG 发射，线程阻塞等数据
            compute(jt, buf);     // 线程还在等 LDG，根本执行不到这里

        在 Ampere 架构之前（Volta/Turing）：没有 cp.async，只能用 “warp 分工”做双缓冲 ———— 生产 warp 搬、消费 warp 算，手动加 barrier

        在 Ampere 架构上，一般双缓冲就是用cp.async
            cp.async  ——>  立即返回（线程继续）  ——>  数据在后台搬
            双缓冲下：
                load_async(jt+1, buf^1);   // cp.async 发射，立即返回
                compute(jt, buf);          // 线程立刻执行，和 tile 1 的搬并行
                wait_group(0);             // 等 tile 1 搬完
                buf ^= 1;
            总时间从 搬 + 算 降到 max(搬, 算)

        Hopper（sm_90）上：用 TMA（cp.async.bulk.tensor） TMA 是 cp.async 的升级版 ———— 一条指令搬整个 tile，不需要 per-thread 循环，更省指令、更省寄存器
    */

    /*
        整个 load_kv 是一个 lambda 函数，为了自动捕获上下文
        [&]是Lambda捕获子句：以引用的方式捕获当前作用域内所有被 lambda 使用的变量
        &：表示所有外部变量（如 K, V, Kbuf, Vbuf, swz, tid, SM_THREADS, HEAD_DIM, TILE_N, S, k_base 等）都通过引用传入 lambda 内部
        捕获之后，这些变量在 lambda 内部就等同于外部原始变量本身，而不是副本
        这里用lambda是为了让 双缓冲复用同一段搬逻辑
    */ 
    auto load_kv = [&](int jtile, int buf)
    {
        // 每个 tile = 32*64*2 = 4096B = 256 个 16B 块
        // 每次 cp.async 搬运 16B = 8个half，所以将整个 tile 划分为 TILE_N × (HEAD_DIM/8) 个 16B chunk
        // cp.async 是 per-thread 指令
        const int CH = TILE_N * (HEAD_DIM / 8);
        for (int c = tid; c < CH; c += SM_THREADS)
        {
            // Chunk 索引
            const int jj = c / (HEAD_DIM / 8);
            const int dd = (c - jj * (HEAD_DIM / 8)) * 8;   // 8 个 half = 16 B
            const int j  = jtile * TILE_N + jj;
            /*
                这里传的是 Chunk 的索引，但还是可以实现swizzle重排
                因为 swz 的粒度就是 16 B，块内 8 个 half 的位置由 | (c & 7) 这一项原样保留下来了
                    swz(r, dd + k) = r*128 + ( ((dd/8) ^ (r&7)) << 3 | k )
                                    = r*128 + ( ((dd/8) ^ (r&7)) << 3 ) + k      // 因为 k < 8，| 等价于 +
                                    = swz(r, dd) + k 
                swz 是 chunk 基址的整体平移 + 块内偏移原样跟随，它把整个 8×16 B 的 chunk 当成一个不可分割的原子来搬
            */ 
            half* ks = Kbuf[buf] + swz(jj, dd);
            half* vs = Vbuf[buf] + swz(jj, dd);
            if (j < S)
            {
                const unsigned ka = (unsigned)__cvta_generic_to_shared(ks);
                const unsigned va = (unsigned)__cvta_generic_to_shared(vs);
                // cp.async.cg.shared.global：Ampere 引入的异步拷贝指令，数据直接从 L2 Cache 到 SMEM，绕过寄存器文件
                // .cg：表示数据缓存在L2
                // 16：固定搬运 16B (= 8 × half = 1 × uint4)
                // 该指令是非阻塞的，发射后线程立即执行下一条指令，实际数据到达由后续的 cp.async.wait_group 保证
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(ka),
                             "l"(K + k_base + (size_t)j * HEAD_DIM + dd));
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(va),
                             "l"(V + k_base + (size_t)j * HEAD_DIM + dd));
            }
            else
            {
                *(uint4*)ks = make_uint4(0u, 0u, 0u, 0u);    // 越界部分补 0，靠 mask 挡
                *(uint4*)vs = make_uint4(0u, 0u, 0u, 0u);
            }
        }
    };

    // 预加载第 0 个 tile 到 buf0
    load_kv(0, 0);
    asm volatile("cp.async.commit_group;\n" ::);        // 把 tile 0 的所有 cp.async 打包成 group 0，提交给硬件队列
    // 此时 tile 0 的搬是“在飞”状态 ———— 数据还没到 smem，但指令已经发出

    for (int jt = 0; jt <= jt_end; ++jt)
    {
        const int buf = jt & 1;                          // 本 tile 用哪个缓冲区

        if (jt < jt_end)    // 最后一个 tile 没有“下一个”，不发射
        {
            // buf ^ 1 异或，单条 XOR 指令，无分支，最常用 (= 1 - buf)
            load_kv(jt + 1, buf ^ 1);       // 先把下一个 tile 发出去
        }
        asm volatile("cp.async.commit_group;\n" ::);
        asm volatile("cp.async.wait_group 1;\n" ::);        // 等“除最后一个 group 外的所有 group”完成；
        // 所以 wait_group 1 保证 本tile的K/V已经搬到smem了，可以读；下一个 tile（jt+1）的搬还在后台，和当前计算重叠

        // 全 block 128 个线程都执行到 wait_group 1 返回；
        // 因为 cp.async 是 per-thread 的，每条线程的 wait_group 只保证“自己发射的 cp.async 完成”，要全 block 的 K/V 都到齐，必须 __syncthreads()
        __syncthreads();
        // 先 wait_group（等自己发射的搬完），再 __syncthreads()（等别人的搬完）
        // 注：mbar_wait就不需要__syncthreads()，因为所有线程 mbar_wait 同一条 barrier(w12f文件)
        
        // 不能是 wait_group 0：等所有 group 完成；  那下一个 tile 的搬也被等，没有重叠——搬和算串行
        // 双缓冲的核心就是 wait_group 1 ———— 留一个 group 在飞，实现重叠

        // warp 级 causal 上界
        // 每个 warp 自己的 causal 上界：只算到自己那 16 行的最后一行（纯省算，不改数值）
        const int jt_end_warp = min(qrow + WARP_M - 1, S - 1) / TILE_N;
        if (jt <= jt_end_warp)
        {
            // S = Q·Kᵀ：4 个 n-tile × 4 个 k 段
            float sc[TILE_N / 8][4];                      // 本 lane 的4 * 4 = 16 个值 
            for (int nt = 0; nt < TILE_N / 8; ++nt)
                for (int c = 0; c < 4; ++c) 
                {
                    sc[nt][c] = 0.f;
                }
            // 4 个 n-tile 段 x 4 个 K 段
            for (int nt = 0; nt < TILE_N / 8; ++nt)
            {
                for (int kk = 0; kk < HEAD_DIM / 16; ++kk)
                {
                    const int row_swz_k = nt * 8 + (lane % 8);
                    const int col_szw_k = kk * 16 + gmask;
                    const half* p = Kbuf[buf] + swz(row_swz_k, col_szw_k);
                    unsigned b0, b1;
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                                : "=r"(b0), "=r"(b1)
                                : "r"((unsigned)__cvta_generic_to_shared(p)));
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                        : "+f"(sc[nt][0]), "+f"(sc[nt][1]), "+f"(sc[nt][2]), "+f"(sc[nt][3])
                        : "r"(aq[kk][0]), "r"(aq[kk][1]), "r"(aq[kk][2]), "r"(aq[kk][3]),
                        "r"(b0), "r"(b1));
                }
            }

            // mask + 行内 max（先在 lane 内，再跨 4 条 lane 合并）
            float mx0 = m0, mx1 = m1;
            for (int nt = 0; nt < TILE_N / 8; ++nt)
            {
                // 本lane的起始全局列(key位置)
                const int j0 = jt * TILE_N + nt * 8 + tcol; 
                for (int c = 0; c < 4; ++c)
                {
                    // c=0 2时 j = j0； c=1 3时，j = j0 + 1 
                    // C布局，c0 c1是相邻列
                    const int j  = j0 + (c & 1);        
                    // c0,c1 属于上半块行 row0，c2,c3 属于下半块行 row1
                    const int rr = (c < 2) ? row0 : row1;
                    float v = sc[nt][c] * scale;
                    if (j > rr || j >= S) v = NEG_INF;
                    sc[nt][c] = v;
                    if (c < 2)
                    {
                        mx0 = fmaxf(mx0, v);
                    }
                    else
                    {
                        mx1 = fmaxf(mx1, v);
                    }
                }
            }
            const float mn0 = warp_max4(mx0);
            const float mn1 = warp_max4(mx1);

            const float a0 = __expf(m0 - mn0);
            const float a1 = __expf(m1 - mn1);
            float s0 = 0.f, s1 = 0.f;
            for (int nt = 0; nt < TILE_N / 8; ++nt)
                for (int c = 0; c < 4; ++c)
                {
                    const float p = __expf(sc[nt][c] - ((c < 2) ? mn0 : mn1));
                    sc[nt][c] = p;
                    if (c < 2) s0 += p;
                    else       s1 += p;
                }
            l0 = l0 * a0 + warp_sum4(s0);
            l1 = l1 * a1 + warp_sum4(s1);
            m0 = mn0; m1 = mn1;

            // O 累加器按 alpha 缩放
            for (int nd = 0; nd < HEAD_DIM / 8; ++nd)
            {
                // c0,c1 属于上半块行 乘 a0
                acc[nd][0] *= a0; acc[nd][1] *= a0;
                acc[nd][2] *= a1; acc[nd][3] *= a1;
            }

            unsigned pA[2][4];
            for (int ks = 0; ks < 2; ++ks)
            {
                pA[ks][0] = pack2(sc[2 * ks + 0][0], sc[2 * ks + 0][1]);
                pA[ks][1] = pack2(sc[2 * ks + 0][2], sc[2 * ks + 0][3]);
                pA[ks][2] = pack2(sc[2 * ks + 1][0], sc[2 * ks + 1][1]);
                pA[ks][3] = pack2(sc[2 * ks + 1][2], sc[2 * ks + 1][3]);
            }

            // O += P·V：2 个 k 段（各 16 个 key）× 8 个 n-tile（每个 8 维）
            for (int ks = 0; ks < 2; ++ks)
            {
                for (int nd = 0; nd < HEAD_DIM / 8; ++nd)
                {          
                    const int vr = ks * 16 + (lane % 8) + ((lane / 8) & 1) * 8;
                    const int col_swz__v = nd * 8;
                    const half* p = Vbuf[buf] + swz(vr, col_swz__v);
                    // const half* p = Vs + (size_t)vr * HEAD_DIM + nd * 8;
                    unsigned b0, b1;
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                                : "=r"(b0), "=r"(b1)
                                : "r"((unsigned)__cvta_generic_to_shared(p)));
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                        : "+f"(acc[nd][0]), "+f"(acc[nd][1]), "+f"(acc[nd][2]), "+f"(acc[nd][3])
                        : "r"(pA[ks][0]), "r"(pA[ks][1]), "r"(pA[ks][2]), "r"(pA[ks][3]),
                        "r"(b0), "r"(b1));
                }
            }
        }  // if (jt <= jt_end_warp)
        __syncthreads();
    }

    // ---- 收尾：只除一次 l ----
    const float inv0 = (l0 > 0.f) ? (1.f / l0) : 0.f;           // l0 是上半块行 row0 的 softmax 分母
    const float inv1 = (l1 > 0.f) ? (1.f / l1) : 0.f;           // l1 是下半块行 row1 的分母
    for (int nd = 0; nd < HEAD_DIM / 8; ++nd)                   // 8 个 D 维段； nd = 0 → D[0..7]     nd = 1 → D[8..15]
    {
        // row0 * HEAD_DIM：上半块行 row0 的偏移
        // nd * 8：第 nd 个 8 列段的起始
        // tcol：本 lane 的列偏移（(lane%4)*2，0/2/4/6）
        half* o0 = O + q_base + (size_t)row0 * HEAD_DIM + nd * 8 + tcol;
        // acc[8][4] 的 8 就是这 8 个 n-tile，nd 循环逐个写回
        o0[0] = __float2half(acc[nd][0] * inv0);            // o0是指针，o0[0] / o0[1] 是它指向的地址和下一个地址
        o0[1] = __float2half(acc[nd][1] * inv0);            // o0[0] 和 o0[1] 是 同一行、相邻两列，地址差 1；每条 lane 只写相邻两列
        if (row1 < S)
        {
            half* o1 = O + q_base + (size_t)row1 * HEAD_DIM + nd * 8 + tcol;
            o1[0] = __float2half(acc[nd][2] * inv1);
            o1[1] = __float2half(acc[nd][3] * inv1);
        }
    }
}


#include <torch/extension.h>

torch::Tensor fa2_mma_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, double scale)
{
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "q/k/v must be CUDA tensors");
    TORCH_CHECK(q.dtype() == torch::kHalf && k.dtype() == torch::kHalf && v.dtype() == torch::kHalf,
                "q/k/v must be fp16");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "q/k/v must be [B,H,S,D]");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
                "q/k/v must be contiguous");

    const int B = q.size(0), H = q.size(1), S = q.size(2), D = q.size(3);
    const int Hkv = k.size(1);
    TORCH_CHECK(D == HEAD_DIM, "head_dim must be 64");
    TORCH_CHECK(S >= 1, "S must be >= 1");
    TORCH_CHECK(H % Hkv == 0, "H must be divisible by Hkv");

    auto out = torch::empty_like(q);
    dim3 grid((S + TILE_M - 1) / TILE_M, B * H);
    fa2_mma_kernel<<<grid, SM_THREADS, SMEM_BYTES>>>(
        reinterpret_cast<const half*>(q.data_ptr()),
        reinterpret_cast<const half*>(k.data_ptr()),
        reinterpret_cast<const half*>(v.data_ptr()),
        reinterpret_cast<half*>(out.data_ptr()),
        B, H, Hkv, S, (float)scale);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "fa2_mma launch failed: ", cudaGetErrorString(err));
    return out;
}
