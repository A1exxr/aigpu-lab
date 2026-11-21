// ===========================================================================
//  w12g_fa2_ws.cu —— FA3 风格的第一版：producer / consumer warp specialization 线程束特化
//  

//  和 w12f 的差别只有「谁干什么」：
//  
//    w12f: 256 线程，8 个 warp 全都既搬（TMA）又算（wgmma + softmax + PV）
//    w12g: 384 线程 = 3 个 warpgroup
//          wg0 = producer：只发 TMA、只等 empty；寄存器 40
//          wg1/wg2 = consumer：只等 full、只算；寄存器 232
//  同步全靠 mbarrier 对：full[buf]（TMA 到齐，tx 计数） / empty[buf]（8 个 warp 都用完）
//
//  判据（本地只需编译 + SASS）：
//    SASS 里必须有 SETMAXNREG（DEC 1 次 INC 1 次）、UTMALDG.3D 2、HGMMA 4、
//    LDGSTS = 0；上机后三档 max|diff| 必须与 w12f 位级一致。

// TMA 和 WGMMA 都是异步硬件单元，将 384 个线程（12 个 Warp / 3 个 Warpgroup）严格划分为 Producer（生产者） 和 Consumer（消费者） 两种角色，通过硬件级同步原语，实现数据搬运与矩阵计算的完美重叠
// ===========================================================================



#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include <cstdio>


#define TILE_M 128              // 一个 block 处理多少行 Q
#define TILE_N 32               // 一次处理多少个 key
#define HEAD_DIM 64
#define SM_THREADS 384          // 3 个 warpgroup：1 producer + 2 consumer
#define WARP_M 16               // PV 阶段：一个 warp 16 行
#define WWG 64                  // QK 阶段：一个 warpgroup 64 行
#define KV_CHUNK 8              // 16 B = 8 half
#define PROD_WARPS 4            // 1 warpgroup = 4 warp
#define CONS_WARPS (SM_THREADS / 32 - PROD_WARPS)
#define NEG_INF (-1e30f)

// ===== mbarrier.arrive（只到、不等）与 setmaxnreg =====
__device__ __forceinline__ void mbar_arrive(uint64_t *bar)
{
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile("mbarrier.arrive.shared.b64 _, [%0];\n" ::"r"(a));
}

// Hopper 支持动态寄存器分配
// setmaxnreg：warpgroup 级别的寄存器再分配（sm_90a 才有）
// Producer 不需要存累加器，主动释放寄存器给 Consumer，使 Consumer 能容纳更大的 Tile 或更多中间变量，避免 Spill
// 必须整个 warpgroup 一起执行；立即数必须是 8 的倍数
__device__ __forceinline__ void wg_reg_dec_producer()
{
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n");
}
__device__ __forceinline__ void wg_reg_inc_consumer()
{
    asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n");
}


// ---------------------------------------------------------------------------
// 基础工具
// ---------------------------------------------------------------------------
__device__ __forceinline__ float h2f(half h) { return __half2float(h); }

__device__ __forceinline__ unsigned pack2(float a, float b)
{
    const half2 h = __floats2half2_rn(a, b);
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

// V 用的 XOR swizzle   128B 的 swizzle 
__device__ __forceinline__ int swz(int r, int c)
{
    return r * HEAD_DIM + ((((c >> 3) ^ (r & 7)) << 3) | (c & 7));
}

__device__ __forceinline__ int cm(int r, int c)
{
    return ((r >> 3) * (HEAD_DIM / 8) + (c >> 3)) * 64 + (r & 7) * 8 + (c & 7);
}

// ---------------------------------------------------------------------------
// wgmma 的 64 位矩阵描述符
//   bit 0-13 : addr >> 4      bit 16-29 : LBO >> 4
//   bit 32-45: SBO  >> 4      bit 62-63 : swizzle（0 = none）
// ---------------------------------------------------------------------------
__device__ __forceinline__ unsigned long long make_desc(const void* p, unsigned lbo,
                                                        unsigned sbo, unsigned swz_mode = 0)
{
    const unsigned long long addr = (unsigned long long)__cvta_generic_to_shared(p);
    unsigned long long d = (addr >> 4) & 0x3FFFu;
    d |= ((unsigned long long)(lbo >> 4) & 0x3FFFu) << 16;
    d |= ((unsigned long long)(sbo >> 4) & 0x3FFFu) << 32;
    d |= ((unsigned long long)(swz_mode & 0x3u)) << 62;
    return d;
}

__device__ __forceinline__ void wgmma_fence()
{
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}
__device__ __forceinline__ void wgmma_commit()
{
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");        // 把前面发出的 mma_async 打包成一个“异步组”，提交到硬件执行队列
}
__device__ __forceinline__ void wgmma_wait0()
{
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
}
 
__device__ __forceinline__ void wgmma_m64n32k16(float (&sc)[TILE_N / 8][4],
                                                unsigned long long a_desc,
                                                unsigned long long b_desc)
{
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, "
        "%16, %17, 1, 1, 1, 0, 0;\n"
        : "+f"(sc[0][0]), "+f"(sc[0][1]), "+f"(sc[0][2]), "+f"(sc[0][3]), "+f"(sc[1][0]), "+f"(sc[1][1]),
          "+f"(sc[1][2]), "+f"(sc[1][3]), "+f"(sc[2][0]), "+f"(sc[2][1]), "+f"(sc[2][2]), "+f"(sc[2][3]),
          "+f"(sc[3][0]), "+f"(sc[3][1]), "+f"(sc[3][2]), "+f"(sc[3][3])
        : "l"(a_desc), "l"(b_desc));
}


// mbarrier
__device__ __forceinline__ void mbar_init(uint64_t *bar, uint32_t count)
{
    uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(a), "r"(count));
}

__device__ __forceinline__ void mbar_arrive_expect_tx(uint64_t *bar, uint32_t bytes)
{
    uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(a), "r"(bytes));
}

__device__ __forceinline__ void mbar_wait(uint64_t *bar, uint32_t phase)
{
    uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile(
    "{\n"
    ".reg .pred p;\n"
    "WAIT_%=:\n"
    "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
    "@!p bra WAIT_%=;\n"
    "}\n" ::"r"(a), "r"(phase));
}

// 3D TMA：把 box [HEAD_DIM, TILE_N, 1] 搬到 smem，坐标 (x=D, y=token, z=bh)
__device__ __forceinline__ void tma_load_3d(void *dst, uint64_t *bar,
                                            const CUtensorMap *tmap,
                                            int x, int y, int z)
{
    uint32_t s = (uint32_t)__cvta_generic_to_shared(dst);
    uint32_t b = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%3, %4, %5}], [%2];\n"
        :: "r"(s), "l"(tmap), "r"(b), "r"(x), "r"(y), "r"(z)
        : "memory");
}


// ---------------------------------------------------------------------------
// kernel
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(SM_THREADS) fa2_ws_kernel(const __grid_constant__ CUtensorMap vmap,
                                                              const __grid_constant__ CUtensorMap kmap,
                                                              const half* __restrict__ Q,
                                                              const half* __restrict__ K,
                                                              const half* __restrict__ V,
                                                              float* __restrict__ O,
                                                              int B, int H, int Hkv, int S, float scale)
{
    // ---- smem ----
    __shared__ alignas(128) half Qs[TILE_M * HEAD_DIM];        // 16 KB core-matrix
    __shared__ alignas(128) half Kbuf[2][TILE_N * HEAD_DIM];   // 2 × 4 KB 128B swizzle
    __shared__ alignas(128) half Vbuf[2][TILE_N * HEAD_DIM];   // 2 × 4 KB swizzle
    __shared__ alignas(8) uint64_t full[2];                    // TMA 到齐（tx 计数）
    __shared__ alignas(8) uint64_t empty[2];                   // 消费者用完了（8 个 warp 各到一次）

    const int bx = blockIdx.x;
    const int bh = blockIdx.y;
    const int b = bh / H;
    const int h = bh % H;
    const int kvh = h / (H / Hkv);

    const int row_base = max(0, min(bx * TILE_M, S - TILE_M));
    const size_t q_base = ((size_t)b * H + h) * S * HEAD_DIM;

    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int warp_id = tid / 32;      // 0..11
    const int wg = warp_id / 4;        // 0 = producer, 1/2 = consumer
    const int wi = warp_id % 4;
    const bool is_prod = (wg == 0);
    const int cw = wg - 1;             // consumer 序号 0/1（producer 下为 -1，不使用）

    const int row0 = cw * WWG + wi * WARP_M + (lane / 4);
    const int row1 = row0 + 8;


    if (tid == 0)
    {
        mbar_init(&full[0], 1);  mbar_init(&full[1], 1);
        mbar_init(&empty[0], CONS_WARPS); mbar_init(&empty[1], CONS_WARPS);
    }
    __syncthreads();

    // ---- Q 进 smem：384 个线程一起搬，谁是 producer 无所谓 ----
    for (int i = tid; i < TILE_M * HEAD_DIM; i += SM_THREADS)
    {
        const int r = i / HEAD_DIM;
        const int d = i % HEAD_DIM;
        const int gr = row_base + r;
        Qs[cm(r, d)] = (gr < S) ? Q[q_base + (size_t)gr * HEAD_DIM + d] : __float2half(0.f);
    }
    __syncthreads();   // 全 block 最后一次同步，之后按角色分开

    const int jt_end = min(row_base + TILE_M - 1, S - 1) / TILE_N;

    if (is_prod)
    {
        // ================= producer：只搬数据，不碰 mma =================
        wg_reg_dec_producer();

        for (int jt = 0; jt <= jt_end; ++jt)
        {
            const int buf = jt & 1;
            if (jt >= 2)
            {
                const int empty_phase = ((jt - 2) >> 1) & 1;
                mbar_wait(&empty[buf], empty_phase);
            }

            if (tid == 0)
            {
                mbar_arrive_expect_tx(&full[buf], 2 * TILE_N * HEAD_DIM * (uint32_t)sizeof(half));
                tma_load_3d(Kbuf[buf], &full[buf], &kmap, 0, jt * TILE_N, bh);
                tma_load_3d(Vbuf[buf], &full[buf], &vmap, 0, jt * TILE_N, bh);
            }
        }
        return;   // producer 到此为止
    }

    // ================= consumer：只算，不搬 =================
    wg_reg_inc_consumer();

    const float qk_scale = scale;   // compute 段里用到的（w12f 里在 kernel 顶部声明）
    float m0 = NEG_INF, m1 = NEG_INF;
    float l0 = 0.f, l1 = 0.f;
    float acc[HEAD_DIM / 8][4];
    for (int nd = 0; nd < HEAD_DIM / 8; ++nd)
        for (int c = 0; c < 4; ++c) acc[nd][c] = 0.f;

    const int jt_end_wg = min(row_base + cw * WWG + WWG - 1, S - 1) / TILE_N;

    for (int jt = 0; jt <= jt_end; ++jt)
    {
        const int buf = jt & 1;
        mbar_wait(&full[buf], (jt >> 1) & 1);   // 等 producer 把这一轮 K/V 搬完

        if (jt <= jt_end_wg)
        {
            float sc[TILE_N / 8][4];
            for (int n = 0; n < TILE_N / 8; ++n)
                for (int c = 0; c < 4; ++c) sc[n][c] = 0.f;

            const unsigned a_lbo = 128, a_sbo = 1024;     // A(Q)：core-matrix，swizzle=0
            const unsigned b_lbo = 1,   b_sbo = 1024;     // B(K)：128B swizzle，LBO 不用、SBO = 8 行 × 128 B

            half* qwg = Qs + cw * (WWG * HEAD_DIM);   // 本 warpgroup 的 64 行
            half* kt = Kbuf[buf];

            wgmma_fence();                            // smem 已就绪
            for (int kk = 0; kk < HEAD_DIM / 16; ++kk)
            {
                const unsigned long long ad = make_desc(qwg + kk * 128, a_lbo, a_sbo, 0);
                // K 的 128B swizzle 布局里，第 kk 个 16-k 段是「行内第 kk 个 32 B」，不是 128 B 的整数倍
                // 这里 K 的描述符 swizzle = 1，描述符里的 swizzle 位必须和 smem 里的物理布局一致
                const unsigned long long bd = make_desc(kt + kk * 16, b_lbo, b_sbo, 1);
                wgmma_m64n32k16(sc, ad, bd);            // 这里是用的 wgmma
            }
            wgmma_commit();
            wgmma_wait0();


            const int gr0 = row_base + row0;
            const int gr1 = row_base + row1;
            float mx0 = NEG_INF, mx1 = NEG_INF;
            for (int n = 0; n < TILE_N / 8; ++n)
            {
                const int j0 = jt * TILE_N + n * 8 + (lane % 4) * 2;
                for (int c = 0; c < 4; ++c)
                {
                    const int j = j0 + (c & 1);
                    const int gr = (c < 2) ? gr0 : gr1;
                    float v = sc[n][c] * qk_scale;
                    if (j > gr || j >= S) v = NEG_INF;
                    sc[n][c] = v;
                    if (c < 2) mx0 = fmaxf(mx0, v);
                    else       mx1 = fmaxf(mx1, v);
                }
            }
            const float mn0 = warp_max4(mx0);
            const float mn1 = warp_max4(mx1);

            const float a0 = __expf(m0 - mn0);
            const float a1 = __expf(m1 - mn1);

            float s0 = 0.f, s1 = 0.f;
            for (int n = 0; n < TILE_N / 8; ++n)
                for (int c = 0; c < 4; ++c)
                {
                    const float p = __expf(sc[n][c] - ((c < 2) ? mn0 : mn1));
                    sc[n][c] = p;
                    if (c < 2) s0 += p;
                    else       s1 += p;
                }
            l0 = l0 * a0 + warp_sum4(s0);
            l1 = l1 * a1 + warp_sum4(s1);
            m0 = mn0; m1 = mn1;

            for (int nd = 0; nd < HEAD_DIM / 8; ++nd)
            {
                acc[nd][0] *= a0; acc[nd][1] *= a0;
                acc[nd][2] *= a1; acc[nd][3] *= a1;
            }

            // ===== 把 softmax 后的 P 打包成 mma.sync 的 A 片段 =====
            unsigned pA[2][4];
            for (int ks = 0; ks < 2; ++ks)
            {
                const int n0 = 2 * ks + 0;;  
                const int n1 = 2 * ks + 1;   
                pA[ks][0] = pack2(sc[n0][0], sc[n0][1]);
                pA[ks][1] = pack2(sc[n0][2], sc[n0][3]);
                pA[ks][2] = pack2(sc[n1][0], sc[n1][1]);
                pA[ks][3] = pack2(sc[n1][2], sc[n1][3]);
            }


            // ---- O += P·V（仍是 mma.sync，V 走 ldmatrix.trans）----
            for (int ks = 0; ks < 2; ++ks)
                for (int nd = 0; nd < HEAD_DIM / 8; ++nd)
                {
                    const int vr = ks * 16 + (lane % 8) + ((lane / 8) & 1) * 8;
                    const half* p = Vbuf[buf] + swz(vr, nd * 8);
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

        if (lane == 0)
            mbar_arrive(&empty[buf]);
    }

    // ---- 收尾：除 l 写回 ----
    const float inv0 = (l0 > 0.f) ? (1.f / l0) : 0.f;
    const float inv1 = (l1 > 0.f) ? (1.f / l1) : 0.f;
    const size_t o_base = ((size_t)b * H + h) * S * HEAD_DIM;
    for (int nd = 0; nd < HEAD_DIM / 8; ++nd)
    {
        const int col = nd * 8 + (lane % 4) * 2;
        if (row_base + row0 < S)
        {
            O[o_base + (size_t)(row_base + row0) * HEAD_DIM + col + 0] = acc[nd][0] * inv0;
            O[o_base + (size_t)(row_base + row0) * HEAD_DIM + col + 1] = acc[nd][1] * inv0;
        }
        if (row_base + row1 < S)
        {
            O[o_base + (size_t)(row_base + row1) * HEAD_DIM + col + 0] = acc[nd][2] * inv1;
            O[o_base + (size_t)(row_base + row1) * HEAD_DIM + col + 1] = acc[nd][3] * inv1;
        }
    }
}

// ---------------------------------------------------------------------------
// host
// ---------------------------------------------------------------------------

// 构建 TMA 描述符，要 3D 的
static int make_tensor_map(CUtensorMap *tmap, void *ptr, int BH, int S, int D)
{
    // 这里是构建V的描述符 V[TILE_N, HEAD_DIM]
    // 在TMA里面x是连续维，y是跳跃维
    cuuint64_t gdim[3]    = {(cuuint64_t)D, (cuuint64_t)S, (cuuint64_t)BH};
    cuuint64_t gstride[2] = {(cuuint64_t)D * sizeof(half),                 // 行跨度 = 128 B
                             (cuuint64_t)D * sizeof(half) * S};            // head 跨度
    cuuint32_t box[3]     = {(cuuint32_t)D, (cuuint32_t)TILE_N, 1};
    cuuint32_t estride[3] = {1, 1, 1};

    CUresult r = cuTensorMapEncodeTiled(
        tmap, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 3, 
        ptr, gdim, gstride,
        box, estride, 
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,                    // smem 里读V用的是128B swizzle
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    if(r != CUDA_SUCCESS)
    {
        const char *msg = "unknow";
        cuGetErrorString(r, &msg);
        printf("cuTensorMapEncodeTiled failed: %s\n", msg);
        return -1;
    }
    return 0;
}

#include <torch/extension.h>
torch::Tensor fa2_wgmma_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, double scale)
{
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "inputs must be CUDA");
    TORCH_CHECK(q.dtype() == torch::kHalf, "q must be fp16");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4, "q/k must be [B,H,S,D]");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(), "must be contiguous");

    const int B = q.size(0), H = q.size(1), S = q.size(2), D = q.size(3);
    const int Hkv = k.size(1);
    TORCH_CHECK(D == HEAD_DIM, "HEAD_DIM must be 64");
    TORCH_CHECK(H % Hkv == 0, "H must be a multiple of Hkv");
    TORCH_CHECK(S >= 1, "S must be >= 1");

    auto o = torch::empty_like(q);
    dim3 grid((S + TILE_M - 1) / TILE_M, B * H);

    CUtensorMap vmap, kmap;
    int bh_total = B * Hkv;
    TORCH_CHECK(make_tensor_map(&vmap, (void*)v.data_ptr(), bh_total, S, HEAD_DIM) == 0, "vmap build failed");
    TORCH_CHECK(make_tensor_map(&kmap, (void*)k.data_ptr(), bh_total, S, HEAD_DIM) == 0, "kmap build failed");

    fa2_ws_kernel<<<grid, SM_THREADS>>>(
        vmap, kmap,
        (const half*)q.data_ptr(), (const half*)k.data_ptr(), (const half*)v.data_ptr(),
        (float*)o.data_ptr(), B, H, Hkv, S, (float)scale);
    const cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "fa2_wgmma launch failed: ", cudaGetErrorString(err));
    return o;
}
