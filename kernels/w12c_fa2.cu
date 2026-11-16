/*
    w12c_fa2.cu —— FA2 前向（causal + GQA）的 mma.sync 完整版（Stage B）

    Stage A（w12c_mma.cu + w12c_qk.py）已经验证：sc[nt][0..3] 就是 S 的 (g,2t),(g,2t+1),(g+8,2t),(g+8,2t+1)。
    那个文件**保留**，它是 QK 的单测：以后改 HEAD_DIM / tile 尺寸 / ldmatrix 地址时先跑它（2 秒出结果），
    能立刻分清「QK 布局错」还是「softmax/PV 错」，不用在本文件里二分。
    本文件在它之上补三件事：
        ① causal mask（j > row 或 j >= S → -inf）
        ② 寄存器内 online softmax（lane 内 4 个数 + 跨 4 条 lane 两次 shuffle）
        ③ P 直接按 A 布局打包 → 16 条 mma 算 P·V → 收尾除 l

    ---- 为什么 P 不用过 smem（FA2 最漂亮的一处）----
      sc[nt] 的 C 布局： c0,c1 = P[g][2t], P[g][2t+1]      c2,c3 = P[g+8][...]
      mma 的 A 布局：    a0,a1 = P[g][2t], P[g][2t+1]      a2,a3 = P[g+8][...]

      **完全一样**。所以 P 的两个 k 段直接打包：
        k 段 0（P 的第 0..15 列）：a0a1←sc[0].c0c1  a2a3←sc[0].c2c3  a4a5←sc[1].c0c1  a6a7←sc[1].c2c3
        k 段 1（P 的第 16..31 列）：同上，用 sc[2]、sc[3]
      → w12b 里那两次 Smem 往返（Ss 存 + Ps 存 + 两次 load_matrix_sync）全部消失。

    ---- 每个 warp 的寄存器预算 ----
      aq  4 段 × 4 = 16 | sc  4 × 4 = 16 | acc 8 × 4 = 32 | pA 2 × 4 = 8 | m/l 4  → 约 76，实测 ~96
      smem 只有 Qs 8 KB + Ks 4 KB + Vs 4 KB = 16 KB  → 可放 5~6 个 block，占用率 ~42%（w12b 是 16%）

    ---- 写法约定 ----
      行：g = lane/4（+8 是下半）；列：2t = (lane%4)*2（+8 跳后半）
      m0/l0 服务行 qrow+g；m1/l1 服务行 qrow+g+8
*/

#include <cuda_fp16.h>

#define TILE_M     64
#define TILE_N     32
#define HEAD_DIM   64
#define SM_THREADS 128
#define WARP_M     16
#define NEG_INF    (-1e30f)
#define SMEM_BYTES (TILE_M * HEAD_DIM * 2 + 2 * TILE_N * HEAD_DIM * 2)

__device__ __forceinline__ unsigned pack2(float a, float b)
{
    const __half2 h = __halves2half2(__float2half_rn(a), __float2half_rn(b));
    return *reinterpret_cast<const unsigned*>(&h);
}

__device__ __forceinline__ float warp_max4(float v)
{
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
    return v;
}

__device__ __forceinline__ float warp_sum4(float v)
{
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    v += __shfl_xor_sync(0xffffffffu, v, 2);
    return v;
}

__global__ void __launch_bounds__(SM_THREADS)
fa2_mma_kernel(const half* __restrict__ Q,
               const half* __restrict__ K,
               const half* __restrict__ V,
               half* __restrict__ O,
               int B, int H, int Hkv, int S, float scale)
{
    extern __shared__ char smem[];
    half* Qs = (half*)smem;                     // [TILE_M][HEAD_DIM]
    half* Ks = Qs + TILE_M * HEAD_DIM;          // [TILE_N][HEAD_DIM]
    half* Vs = Ks + TILE_N * HEAD_DIM;          // [TILE_N][HEAD_DIM]

    const int bh   = blockIdx.y;
    const int b    = bh / H;
    const int h    = bh - b * H;
    const int grp  = H / Hkv;
    const int kvh  = h / grp;

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;
    const int g       = lane >> 2;
    const int tcol    = (lane & 3) * 2;
    const int gmask   = ((lane / 8) & 1) * 8;   // ldmatrix 的列组偏移

    // S >= TILE_M 时把块起点夹到 S-TILE_M，保证块内 64 行全部真实存在（w12b 尾块 bug 的修法）
    const int row_base = min(blockIdx.x * TILE_M, S - TILE_M);
    const int qrow     = row_base + warp_id * WARP_M;

    const size_t q_base = ((size_t)b * H   + h  ) * S * HEAD_DIM;
    const size_t k_base = ((size_t)b * Hkv + kvh) * S * HEAD_DIM;

    // ---- ① Q 进 smem ----
    for (int i = tid; i < TILE_M * HEAD_DIM; i += SM_THREADS)
    {
        const int r = i / HEAD_DIM, d = i - r * HEAD_DIM;
        Qs[i] = Q[q_base + (size_t)(row_base + r) * HEAD_DIM + d];
    }
    __syncthreads();

    // ---- ② Q → 4 个 A 片段 ----
    unsigned aq[HEAD_DIM / 16][4];
    for (int kk = 0; kk < HEAD_DIM / 16; ++kk)
    {
        const half* p = Qs + (size_t)(warp_id * WARP_M + (lane % 16)) * HEAD_DIM
                           + kk * 16 + (lane / 16) * 8;
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                     : "=r"(aq[kk][0]), "=r"(aq[kk][1]), "=r"(aq[kk][2]), "=r"(aq[kk][3])
                     : "r"((unsigned)__cvta_generic_to_shared(p)));
    }

    // ---- ③ online softmax 状态 + O 累加器（全在寄存器）----
    float m0 = NEG_INF, m1 = NEG_INF, l0 = 0.f, l1 = 0.f;
    float acc[HEAD_DIM / 8][4];
    for (int nd = 0; nd < HEAD_DIM / 8; ++nd)
        for (int c = 0; c < 4; ++c) acc[nd][c] = 0.f;

    const int row0  = qrow + g;
    const int row1  = qrow + g + 8;
    const int jt_end = min(row_base + TILE_M - 1, S - 1) / TILE_N;

    for (int jt = 0; jt <= jt_end; ++jt)
    {
        // ④ K/V tile 进 smem（越界填 0，靠 mask 挡）
        for (int i = tid; i < TILE_N * HEAD_DIM; i += SM_THREADS)
        {
            const int jj = i / HEAD_DIM, dd = i - jj * HEAD_DIM;
            const int j  = jt * TILE_N + jj;
            const bool ok = (j < S);
            Ks[i] = ok ? K[k_base + (size_t)j * HEAD_DIM + dd] : __float2half(0.f);
            Vs[i] = ok ? V[k_base + (size_t)j * HEAD_DIM + dd] : __float2half(0.f);
        }
        __syncthreads();

        // ⑤ S = Q·Kᵀ：4 个 n-tile × 4 个 k 段
        float sc[TILE_N / 8][4];
        for (int nt = 0; nt < TILE_N / 8; ++nt)
            for (int c = 0; c < 4; ++c) sc[nt][c] = 0.f;

        for (int nt = 0; nt < TILE_N / 8; ++nt)
        {
            for (int kk = 0; kk < HEAD_DIM / 16; ++kk)
            {
                const half* p = Ks + (size_t)(nt * 8 + (lane % 8)) * HEAD_DIM
                                   + kk * 16 + gmask;
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

        // ⑥ mask + 行内 max（先在 lane 内，再跨 4 条 lane 合并）
        float mx0 = m0, mx1 = m1;
        for (int nt = 0; nt < TILE_N / 8; ++nt)
        {
            const int j0 = jt * TILE_N + nt * 8 + tcol;
            for (int c = 0; c < 4; ++c)
            {
                const int j  = j0 + (c & 1);
                const int rr = (c < 2) ? row0 : row1;
                float v = sc[nt][c] * scale;
                if (j > rr || j >= S) v = NEG_INF;
                sc[nt][c] = v;
                if (c < 2) mx0 = fmaxf(mx0, v);
                else       mx1 = fmaxf(mx1, v);
            }
        }
        const float mn0 = warp_max4(mx0);
        const float mn1 = warp_max4(mx1);

        const float a0 = __expf(m0 - mn0);       // 首轮 m0=-inf → 0
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

        // ⑦ O 累加器按 alpha 缩放（寄存器里，32 次乘法就完事）
        for (int nd = 0; nd < HEAD_DIM / 8; ++nd)
        {
            acc[nd][0] *= a0; acc[nd][1] *= a0;
            acc[nd][2] *= a1; acc[nd][3] *= a1;
        }

        // ⑧ P → A 布局（零 smem）：k 段 0 用 sc[0],sc[1]；k 段 1 用 sc[2],sc[3]
        unsigned pA[2][4];
        for (int ks = 0; ks < 2; ++ks)
        {
            pA[ks][0] = pack2(sc[2 * ks + 0][0], sc[2 * ks + 0][1]);
            pA[ks][1] = pack2(sc[2 * ks + 0][2], sc[2 * ks + 0][3]);
            pA[ks][2] = pack2(sc[2 * ks + 1][0], sc[2 * ks + 1][1]);
            pA[ks][3] = pack2(sc[2 * ks + 1][2], sc[2 * ks + 1][3]);
        }

        // ⑨ O += P·V：2 个 k 段（各 16 个 key）× 8 个 n-tile（每个 8 维）
        for (int ks = 0; ks < 2; ++ks)
        {
            for (int nd = 0; nd < HEAD_DIM / 8; ++nd)
            {
                // V 是 [key][dim]；mma 要 B = Vᵀ 的 col 布局 → 用 .trans
                const int vr = ks * 16 + (lane % 8) + ((lane / 8) & 1) * 8;
                const half* p = Vs + (size_t)vr * HEAD_DIM + nd * 8;
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
        __syncthreads();
    }

    // ---- ⑩ 收尾：只除一次 l ----
    const float inv0 = (l0 > 0.f) ? (1.f / l0) : 0.f;
    const float inv1 = (l1 > 0.f) ? (1.f / l1) : 0.f;
    for (int nd = 0; nd < HEAD_DIM / 8; ++nd)
    {
        const int d = nd * 8 + tcol;
        half* o0 = O + q_base + (size_t)row0 * HEAD_DIM + d;
        o0[0] = __float2half(acc[nd][0] * inv0);
        o0[1] = __float2half(acc[nd][1] * inv0);
        if (row1 < S)
        {
            half* o1 = O + q_base + (size_t)row1 * HEAD_DIM + d;
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
    TORCH_CHECK(S >= TILE_M, "S must be >= 64");
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
