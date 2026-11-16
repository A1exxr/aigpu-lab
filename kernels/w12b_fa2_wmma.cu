/*
    w12b_fa2_wmma.cu — FA2 前向（causal + GQA）的 Tensor Core 版（wmma API）
    与 w12_fa2.cu 的差别只有一处：
         标量 FFMA 算点积(每个 score 是一个线程用 32 次 FFMA 算出来的) ————> wmma 的 16x16x16 矩阵乘
         每元素一次 LDS，1 LDS : 1 FMA ————> 一个 fragment 一次装 16×16，硬件内部复用
         每元素 h2f() 一次  ————> mma 内部 fp16 输入 / fp32 累加，省掉逐元素转换指令

    代价：wmma 的累加器是 fragment（不透明寄存器块），不能按行索引访问(frag[i][j])、不能按行缩放
        所以所有需要“按行操作”的东西（mask、softmax、alpha rescale）都必须先把 fragment 存到 smem，在smem上逐元素处理完再装回去
        这就是这份代码 smem 用量大、occupancy 只有 16% 的原因
    

    结构（Q 外层并行 / causal 跳块 / GQA / online softmax）与 w12 完全一致

    ---- 数据流（每个 K/V tile）----
     全局 K/V ──► Ks/Vs (smem, fp16)
                    │
                    ▼
            S = Q·Kᵀ (wmma, fp32 fragment)
                    │ store
                    ▼
            Ss (smem, fp32) ──► 逐元素: mask / m_new / alpha / p
                    │                        │
                    │                        ▼
                    │                   Ps (smem, fp16)
                    │                        │
                    ▼                        ▼
            Acc (smem, fp32) ◄── O = P·V (wmma, 经 Ss 临时缓冲)
                    │
                    ▼
            写回 O = Acc / l

    ---- smem 预算（bytes）----
      Ks 4096 | Vs 4096 | Ss 8192 | Ps 4096 | Acc 16384 | m/l/alpha 768 = 37632  (36.75 KB)
      每 SM 可放 2 个 block  占用率只有 16%（8 warp / 48），这是 wmma 版的天花板
      根因就是 Acc 必须过 smem（wmma 的累加器不能按行缩放） w12c 把 Acc 放回寄存器才能解决

*/


#include <cuda_fp16.h>
#include <mma.h>

#define TILE_M     64      
#define TILE_N     32      
#define HEAD_DIM   64
#define SM_THREADS 128     
#define NEG_INF    (-1e30f)
using namespace nvcuda;

__global__ void __launch_bounds__(SM_THREADS) fa2_wmma_kernel(const half* __restrict__ Q, 
                                                            const half* __restrict__ K,
                                                            const half* __restrict__ V, 
                                                            half* __restrict__ O,
                                                            int B, int H, int Hkv, int S, float scale)
{
    // S=512, B=2, H=8, Hkv=2, D=64
    extern __shared__ char smem[];
    half* Ks = (half*)smem;                                 // [TILE_N][HEAD_DIM]
    half* Vs = Ks + TILE_N * HEAD_DIM;                      // [TILE_N][HEAD_DIM]
    float* Ss = (float*)(Vs + TILE_N * HEAD_DIM);           // [TILE_M][TILE_N]，也当 PV 的临时缓冲
    half* Ps = (half*)(Ss + TILE_M * TILE_N);               // [TILE_M][TILE_N]
    float* Acc = (float*)(Ps + TILE_M * TILE_N);            // [TILE_M][HEAD_DIM]

    // 形状是一个warp 32线程 共同处理fragment块，一个block有4个warp，m l alpha需要放到smem里
    float* mrow = Acc + TILE_M * HEAD_DIM;                  // [TILE_M]     mrow[r] 第 r 行 Q 的 running max
    float* lrow = mrow + TILE_M;                            // [TILE_M]     lrow[r]	第 r 行 Q 的 softmax 分母累积
    float* arow = lrow + TILE_M;                            // [TILE_M]     arow[r]	第 r 行 Q 本轮的缩放因子 alpha

    const int bh = blockIdx.y;
    const int b  = bh / H;
    const int h  = bh - b * H;
    const int grp = H / Hkv;                            // GQA：几个 q head 共用一个 kv head
    const int kvh = h / grp;

    // grid(8, 2*8)
    const size_t q_base = ((size_t)b * H   + h  ) * S * HEAD_DIM;
    const size_t k_base = ((size_t)b * Hkv + kvh) * S * HEAD_DIM;

    const int tid  = threadIdx.x;
    const int warp_id = tid >> 5;                          // 0..3
    const int lane = tid & 31;

    // 不再是两个 lane 合作 1行Q，而是通过 wmma 算
    const int row_base = (S >= TILE_M) ? min(blockIdx.x * TILE_M, S - TILE_M) : blockIdx.x * TILE_M;
    // 本 warp 负责的 16 行 Q的起点
    const int row_w = row_base + warp_id * 16;


    // ---- 1. Q 装进 fragment（整个 block 生命周期只读一次 Q）----
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> aq[HEAD_DIM / 16];
    // D：64维，64/16=4，有4个段
    // 一个 warp 装 [16,64] 的 Q，一个 block 负责的 Q 是 [64,64] ————  aq[4] 合起来才是 16 * 64 的 Q
    for (int kk = 0; kk < HEAD_DIM / 16; ++kk)
    {   
        // 装 Q 的第 kk 段 16 维（A 片段，行主序）
        wmma::load_matrix_sync(aq[kk], Q + q_base + (size_t)row_w * HEAD_DIM + kk * 16, HEAD_DIM);
    }

    // ---- 2. 初始化 ----
    for (int i = tid; i < TILE_M * HEAD_DIM; i += SM_THREADS) Acc[i] = 0.f;
    for (int r = tid; r < TILE_M; r += SM_THREADS) 
    { 
        mrow[r] = NEG_INF; 
        lrow[r] = 0.f; 
        arow[r] = 1.f; 
    }
    __syncthreads();

    // ---- 3. 主循环：只走到 块内最后一行 所在的 K/V tile ----
    const int row_last = min(row_base + TILE_M - 1, S - 1);
    const int jt_end   = row_last / TILE_N;

    for (int jt = 0; jt <= jt_end; ++jt)
    {
        // 3.1 K/V tile 进 smem（越界填 0，反正会被 mask 掉）
        for (int idx = tid; idx < TILE_N * HEAD_DIM; idx += SM_THREADS)
        {
            const int jj = idx / HEAD_DIM;
            const int dd = idx - jj * HEAD_DIM;
            const int j  = jt * TILE_N + jj;
            const bool ok = (j < S);
            Ks[idx] = ok ? K[k_base + (size_t)j * HEAD_DIM + dd] : __float2half(0.f);           // Ks形状[32,64]
            Vs[idx] = ok ? V[k_base + (size_t)j * HEAD_DIM + dd] : __float2half(0.f);
        }
        __syncthreads();

        // 3.2 S 片段 = Q · Kᵀ
        // wmma的fragment是16*16*16，所以三个维度都要看拆不拆
        // S[16][32] = Q[16][64] · Ksᵀ[64][32]
        // Q 的 M 维是16，不用拆；K是归约维，拆4段，但累加，不新建fragment；N 是 32，要拆成2个，就是sc[2]
        // sc 是 accumulator 输出累加器，沿 N 拆，输出的S(也就是C的一部分) 形状是 M*N [16,32]，所以沿着 N 拆 N/16=2个
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> sc[TILE_N / 16];
        for (int ng = 0; ng < TILE_N / 16; ++ng) 
        {
            wmma::fill_fragment(sc[ng], 0.f);
        }

        // 算 S 片段的时候，就是NT变体；在存Ks的时候，是按行主序访问存的，所以这个片段声明就用col_major
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> bk;
        for (int ng = 0; ng < TILE_N / 16; ++ng)            // N 方向
        {
            for (int kk = 0; kk < HEAD_DIM / 16; ++kk)      // K方向
            {
                // col_major 让 逻辑上(row, col)的矩阵，在加载时按照 col * ld + row 来加载
                // 所以下面就是 N方向 * ld + K方向(也就是row) 
                // 读“逻辑 B 的 (k,n)”，实际读的是“物理 Ks 的 (n,k)”——正好是 Kᵀ
                wmma::load_matrix_sync(bk, Ks + (ng * 16) * HEAD_DIM + kk * 16, HEAD_DIM);

                // 累加
                wmma::mma_sync(sc[ng], aq[kk], bk, sc[ng]);
            }
        }
        for (int ng = 0; ng < TILE_N / 16; ++ng)
        {
            wmma::store_matrix_sync(Ss + (warp_id * 16) * TILE_N + ng * 16, sc[ng], TILE_N, wmma::mem_row_major);
        }
        __syncthreads();

        // 3.3 行处理：mask → m_new → alpha → p，同时把 Acc 按 alpha 缩放
        /*
            w12b 虽然用了 Tensor Core，但 wmma::fragment 是黑盒，不知道 x[0] 对应 C 的哪一行哪一列，所以 mask/softmax/rescale 只能写回 smem 逐元素做，每 tile 一次 16 KB 的 smem 往返

            通过用 mma.sync，指令集规定了： A/B/C 的寄存器布局（a0,a1 = A[lane/4][(lane%4)*2+0/1] 等），照着摆就能在寄存器里直接做 mask/softmax/rescale，不碰 smem

            wmma 的 16×16 在 Ampere 上通常由两条 m16n8k16 拼成（N 方向拆），但这是实现细节，不是 ISA 保证；m16n8k16 的布局是 PTX 写死的

            ldmatrix 负责把数据从 smem 搬到寄存器，每条 lane 给一个行首，硬件连续搬 8 个 half（16 字节 = 128 bit = .m8n8 的一行），32 lane × 16 字节铺满 4 个 8×8 块

            A 要“行内相邻”，A 本来就是行主序，直接存、直接搬；
            B 要“列内相邻”，B 行主序时“同一列相邻行”差 8、不连续，所以在装 smem 时转置成 BsT[n][k]，让“同列相邻行”变成“同行相邻列”，ldmatrix 才能连续搬。转置发生在装填，不在访问。
        */
        for (int r = tid; r < TILE_M; r += SM_THREADS)
        {
            const int row = row_base + r;
            if (row >= S) 
            {
                mrow[r] = NEG_INF; 
                lrow[r] = 0.f; 
                arow[r] = 1.f; 
                continue; 
            }

            float mx = mrow[r];
            for (int c = 0; c < TILE_N; ++c)
            {
                const int j = jt * TILE_N + c;
                float s = Ss[r * TILE_N + c] * scale;
                if (j > row || j >= S) s = NEG_INF;
                Ss[r * TILE_N + c] = s;
                mx = fmaxf(mx, s);
            }

            const float m_new = mx;
            const float alpha = __expf(mrow[r] - m_new);      // 首次 m=-inf → alpha=0
            float l = lrow[r] * alpha;
            for (int c = 0; c < TILE_N; ++c)
            {
                const float p = __expf(Ss[r * TILE_N + c] - m_new);
                Ps[r * TILE_N + c] = __float2half(p);
                l += p;
            }
            mrow[r] = m_new;
            lrow[r] = l;
            arow[r] = alpha;
        }
        __syncthreads();

        for (int i = tid; i < TILE_M * HEAD_DIM; i += SM_THREADS)
        {
            Acc[i] *= arow[i / HEAD_DIM];
        }
        __syncthreads();

        // 3.4 O 片段 = P · V  ———— O[64][64] = P[64][TILE_N=32] · V[TILE_N=32][64]
        // D维 分两半做（各 32 维 = 2 个 16x16 片段），Ss 当临时缓冲
        // 分成[64,0...31]  和 [64, 32...63] 
        /*
            O = P·V 
                P: [M=16][K=32]      ← 16 行 Q，32 个 key
                V: [K=32][N=64]      ← 32 个 key，64 维 D
                O: [M=16][N=64]      ← 16 行 Q，64 维输出
            每个 warp 负责 16 行，输出 O_warp[16][64]  ————  4 个 warp 拼成 O[64][64]
            half 循环切的是 O_warp 的 N 维（64 列 D），切成两个 32 列
                每个 half 里：oc[2] 覆盖 32 列 ———— O[16][32]
                一个 fragment 是 16×16，oc[0] 管 16 列，oc[1] 管另外 16 列，合起来 32 列 = 一个 half
                算O[16][32]时，P是[16][32]，V是[32][64]，但这里V只用到了32列，一个half只用到32列
                先把 P 的列拆成两个16  ———— kk循环
                    V 的行拆成两个 16 ———— 和 P 的列是同一个 kk，因为 V 的行是K维，P 的列也是K维，两个是同一个K维
                        V 的列拆成 4个16，但每个half只用2个，通过 half 和 t 两层循环选段
                        然后就是一直用 16*16*16 进行计算
        */
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> ap;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bv;
        
        for (int half = 0; half < 2; ++half)
        {
            // 这里一个half 32 列，拆成两个16列段，所以是两个累加器
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> oc[2];
            wmma::fill_fragment(oc[0], 0.f);
            wmma::fill_fragment(oc[1], 0.f);

            for (int kk = 0; kk < TILE_N / 16; ++kk)
            {
                // 装 P 的第 kk 段（A 片段，行主序）
                wmma::load_matrix_sync(ap, Ps + (warp_id * 16) * TILE_N + kk * 16, TILE_N);

                for (int t = 0; t < 2; ++t)
                {
                    const int dg = half * 2 + t;          // 0..3，对应 D 的 4 个 16 维段
                    // half=0, t=0 → dg=0 → V[:, 0..15]
                    // half=0, t=1 → dg=1 → V[:, 16..31]
                    // half=1, t=0 → dg=2 → V[:, 32..47]
                    // half=1, t=1 → dg=3 → V[:, 48..63]

                    // 装 V 的第 kk 个 16 token × 第 dg 个 16 维（B 片段，行主序）
                    wmma::load_matrix_sync(bv, Vs + (kk * 16) * HEAD_DIM + dg * 16, HEAD_DIM);
                    
                    wmma::mma_sync(oc[t], ap, bv, oc[t]);
                }
            }
            // 4 个 warp 各写 16x32 的 fp32 条带，正好铺满 Ss[64][32]
            for (int t = 0; t < 2; ++t)
            {
                wmma::store_matrix_sync(Ss + (warp_id * 16) * TILE_N + t * 16, oc[t], TILE_N, wmma::mem_row_major);
            }
            __syncthreads();

            for (int i = tid; i < TILE_M * TILE_N; i += SM_THREADS)
            {
                const int r = i / TILE_N;
                const int c = i - r * TILE_N;
                Acc[r * HEAD_DIM + half * TILE_N + c] += Ss[i];
            }
            __syncthreads();
        }
    }

    // ---- 4. 写回：只在最后除一次 l ----
    for (int r = tid; r < TILE_M; r += SM_THREADS)
    {
        const int row = row_base + r;
        if (row >= S) continue;
        const float inv = (lrow[r] > 0.f) ? (1.f / lrow[r]) : 0.f;
        for (int d = 0; d < HEAD_DIM; ++d)
            O[q_base + (size_t)row * HEAD_DIM + d] = __float2half(Acc[r * HEAD_DIM + d] * inv);
    }
}


#include <torch/extension.h>

#define SMEM_BYTES (TILE_N * HEAD_DIM * 2 * 2      /* Ks + Vs */ \
                  + TILE_M * TILE_N * 4            /* Ss       */ \
                  + TILE_M * TILE_N * 2            /* Ps       */ \
                  + TILE_M * HEAD_DIM * 4          /* Acc      */ \
                  + TILE_M * 4 * 3)                /* m/l/alpha */

torch::Tensor fa2_wmma_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, double scale)
{
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(q.dtype() == torch::kHalf, "q must be fp16");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4, "q/k must be [B, H, S, D]");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(), "inputs must be contiguous");

    const int B = q.size(0), H = q.size(1), S = q.size(2), D = q.size(3);
    const int Hkv = k.size(1);
    TORCH_CHECK(D == HEAD_DIM, "head_dim must be 64");
    TORCH_CHECK(H % Hkv == 0, "H must be divisible by Hkv");

    auto out = torch::empty_like(q);
    dim3 grid((S + TILE_M - 1) / TILE_M, B * H);
    dim3 block(SM_THREADS);
    fa2_wmma_kernel<<<grid, block, SMEM_BYTES>>>(
        (const half*)q.data_ptr(), (const half*)k.data_ptr(),
        (const half*)v.data_ptr(), (half*)out.data_ptr(),
        B, H, Hkv, S, (float)scale);
    return out;
}
