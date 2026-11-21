/*
    w12e_fa2_wgmma.cu —— FA2 前向（causal + GQA），QK 用 wgmma，PV 仍用 mma.sync

    K 用 wgmma，PV 仍用 mma.sync

    注：
        1、wmma 是 Ampere 之前的 CUDA C++ API（nvcuda::wmma::fragment），w12b 用的是这个
        2、mma.sync 是 PTX 指令（warp 级、同步），w12c/w12d/w12e 的 PV 用的是这个
        3、wgmma 是 Hopper 的 warpgroup 级、异步指令，w12e 的 QK 用的是这个


    与 w12d_fa2_tile128.cu 的三处结构差别：
      1. Q/K 的 smem 布局从 XOR swizzle 改为 core-matrix 顺序（wgmma）
         V 仍用 swizzle 布局（PV 还是 mma.sync + ldmatrix）
      2. QK 从 4×mma.sync.m16n8k16 换成 4×wgmma.m64n32k16，
         线程划分从 1 warp = 16 行 ————> 变成  1 warpgroup = 64 行 
      3. 同步从 __syncthreads() 变成 wgmma_fence/commit/wait0（wgmma 是异步的）

    只有 sm_90a 能编出 HGMMA：
      nvcc -arch=sm_90a -cubin w12e_fa2_wgmma.cu
    本地 4060（sm_89）编不了，数值要等 H800
*/


#include <cuda_fp16.h>
#include <cuda_runtime.h>


#define TILE_M 128              // 一个 block 处理多少行 Q
#define TILE_N 32               // 一次处理多少个 key
#define HEAD_DIM 64
#define SM_THREADS 256          // = 2 个 warpgroup
#define WARP_M 16               // PV 阶段：一个 warp 16 行
#define WWG 64                  // QK 阶段：一个 warpgroup 64 行
#define KV_CHUNK 8              // 16 B = 8 half
#define NEG_INF (-1e30f)

// ---------------------------------------------------------------------------
// 基础工具
// ---------------------------------------------------------------------------
__device__ __forceinline__ float h2f(half h) { return __half2float(h); }

__device__ __forceinline__ unsigned pack2(float a, float b)
{
    const half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<const unsigned*>(&h);
}

// 4 条 lane 一起做行内归约（这 4 条 lane 共享同一行）
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

// Q/K 的形状都是 [M,K]，行主序，K维连续，块内偏移都是“行偏移 × 8 + K 维偏移”，所以公式一样
// Q/K 用的 core-matrix 下标：块号 (r/8)*(HEAD_DIM/8) + (c/8)，块内 (r%8)*8 + (c%8)
//   r: 行（Q 的 m / K 的 key），c: 列（HEAD_DIM 方向，必须 8 对齐地成块）
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
    // __cvta_generic_to_shared 返回的是 size_t(64位)
    // 描述符本身就是 64位 的，所以这里就需要 64位
    // 而之前的cp.async的地址需要32位操作数，所以要截断
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
 
// m64n32k16：每个线程 16 个 f32 累加器   64*32/128=16
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

// cp.async：ok=false 时 zfill（越界的 key 读成 0，靠 mask 挡住）
__device__ __forceinline__ void cp16(half* dst, const half* src, bool ok)
{
    // PTX 的 .shared 操作数都要 32 位 shared 地址
    // [%0]：目标 smem 地址
    // [%1]：源 global 地址
    // 16：搬运字节数
    // %2：src-size（zfill 大小）
    // src-size 是“可选操作数”——不写就是“搬满 16 字节”，写了就是“搬 src_size 字节 + 填 0”
    // 语义：从 src 搬 src_size 字节到 dst，剩余 (16-src_size）字节填 0
    const unsigned s = (unsigned)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(src),
                 "r"(ok ? 16 : 0));
}
// commit_group 把“自上次 commit 以来发射的所有 cp.async”打包成一个 group，提交给硬件队列
// 实现 批量等待
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }

// wait_group N：等“队列里剩下 N 个 group 未完成时”返回
// 队列里有 3 个 group（g0, g1, g2）
//      wait_group 0：等所有 3 个完成（剩 0 个）
//      wait_group 1：等 g0、g1 完成（g2 还在飞，剩 1 个）
//      wait_group 2：等 g0 完成（g1、g2 还在飞，剩 2 个）
// 这里就是指 等“除最后一个外的所有 group”
__device__ __forceinline__ void cp_wait1() { asm volatile("cp.async.wait_group 1;\n" ::: "memory"); }

// ---------------------------------------------------------------------------
// kernel
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(SM_THREADS) fa2_wgmma_kernel(const half* __restrict__ Q, 
                                                                const half* __restrict__ K, 
                                                                const half* __restrict__ V,
                                                                float* __restrict__ O, 
                                                                int B, int H, int Hkv, int S, float scale)
{
    __shared__ alignas(128) half Qs[TILE_M * HEAD_DIM];         // 16 KB，core-matrix
    __shared__ alignas(128) half Kbuf[2][TILE_N * HEAD_DIM];    // 2 × 4 KB，core-matrix
    __shared__ alignas(128) half Vbuf[2][TILE_N * HEAD_DIM];    // 2 × 4 KB，swizzle

    const int bx = blockIdx.x;
    const int bh = blockIdx.y;
    const int b = bh / H;
    const int h = bh % H;
    const int kvh = h / (H / Hkv);

    const int row_base = max(0, min(bx * TILE_M, S - TILE_M));

    const size_t q_base = ((size_t)b * H + h) * S * HEAD_DIM;
    const size_t kv_base = ((size_t)b * Hkv + kvh) * S * HEAD_DIM;

    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int warp_id = tid / 32;       // 0..7
    const int wg = warp_id / 4;         // 第几个 warpgroup：0/1
    const int wi = warp_id % 4;         // warpgroup 内第几个 warp：0..3

    // 因为 mma.sync/wgmma 的 C 布局，所以是4个lane管tile的一行
    // 本线程负责的两行（tile 内下标）
    const int row0 = wg * WWG + wi * WARP_M + (lane / 4);           //  lane / 4	0..7	warp 内第几行（4 条 lane 一组管一行）
    const int row1 = row0 + 8;                                      // 同一个lane   +8 行

    // ---- Q 进 smem（core-matrix 顺序）----
    for (int i = tid; i < TILE_M * HEAD_DIM; i += SM_THREADS)
    {
        const int r = i / HEAD_DIM;
        const int d = i % HEAD_DIM;
        const int gr = row_base + r;
        Qs[cm(r, d)] = (gr < S) ? Q[q_base + (size_t)(gr) * HEAD_DIM + d] : __float2half(0.f);
    }

    // ---- K/V 装载 lambda ----
    auto load_kv = [&](int jtile, int buf)
    {
        for (int i = tid; i < TILE_N * (HEAD_DIM / KV_CHUNK); i += SM_THREADS)
        {
            const int jj = i / (HEAD_DIM / KV_CHUNK);
            const int dd = (i % (HEAD_DIM / KV_CHUNK)) * KV_CHUNK;          // 这里 dd 是8的倍数，cm 和 swz 都接受 8 对齐的 dd
            const int key = jtile * TILE_N + jj;
            const bool ok = (key < S);
            const half* pk = K + kv_base + (size_t)key * HEAD_DIM + dd;
            const half* pv = V + kv_base + (size_t)key * HEAD_DIM + dd;
            cp16(Kbuf[buf] + cm(jj, dd), pk, ok);                           // K 用 core-matrix； 这里没有用TMA来搬，是接下来的优化地方
            cp16(Vbuf[buf] + swz(jj, dd), pv, ok);                          // V 用 swizzle
        }
    };

    // ---- online softmax 状态 ----
    const float qk_scale = scale;
    float m0 = NEG_INF, m1 = NEG_INF;
    float l0 = 0.f, l1 = 0.f;
    float acc[HEAD_DIM / 8][4];
    for (int nd = 0; nd < HEAD_DIM / 8; ++nd)
        for (int c = 0; c < 4; ++c) acc[nd][c] = 0.f;

    const int jt_end = min(row_base + TILE_M - 1, S - 1) / TILE_N;              // block 级
    const int jt_end_wg = min(row_base + wg * WWG + WWG - 1, S - 1) / TILE_N;   // 本 warpgroup

    // ---- 双缓冲 prologue ----
    load_kv(0, 0);
    cp_commit();

    int buf = 0;
    for (int jt = 0; jt <= jt_end; ++jt)
    {
        if (jt + 1 <= jt_end)
            load_kv(jt + 1, buf ^ 1);
        cp_commit();
        cp_wait1();
        __syncthreads();   // Kbuf[buf] 到齐

        if (jt <= jt_end_wg)
        {
            // ---- S = Q·Kᵀ：4 个 k 段 ----
            float sc[TILE_N / 8][4];
            for (int n = 0; n < TILE_N / 8; ++n)
                for (int c = 0; c < 4; ++c) sc[n][c] = 0.f;

            // wgmma 的 A/B 是 core matrix 阵列，硬件靠 4 个数去跳块：
            // cm(r, c) = ((r>>3)*(HEAD_DIM/8) + (c>>3))*64 + (r&7)*8 + (c&7)
            //   LBO = 沿 块内快轴 走 1 块要跨多少字节
            //   SBO = 沿 块号慢轴 走 1 块要跨多少字节
            // 每个 core matrix = 8×8 = 64 half = 128 字节
            // 沿 K 方向（c 增加 8）：块号 +1 ——> 字节偏移 +128
            // 沿 M/N 方向（r 增加 8）：块号 +(HEAD_DIM/8) = +8 ——> 字节偏移 +8×128 = 1024
            const unsigned a_lbo = 128, a_sbo = 1024;
            const unsigned b_lbo = 128, b_sbo = 1024;

            half* qwg = Qs + wg * (WWG * HEAD_DIM);   // 本 warpgroup 的 64 行
            half* kt = Kbuf[buf];

            wgmma_fence();                            // smem 已就绪
            for (int kk = 0; kk < HEAD_DIM / 16; ++kk)
            {
                // kk 段 → 跳过 kk*16 列 = kk*2 个 k 块 = kk*128 个 half
                const unsigned long long ad = make_desc(qwg + kk * 128, a_lbo, a_sbo, 0);
                const unsigned long long bd = make_desc(kt + kk * 128, b_lbo, b_sbo, 0);
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
            // wgmma 的累加器布局和 mma.sync 的 C 布局一模一样，所以也可以打包
            // sc[n][0],sc[n][1] 是 row0 的两列；sc[n][2],sc[n][3] 是 row1 的两列
            // PV 的 mma.sync 是 m16n8k16：A 片段每条 lane 要 8 个 half = 4 个 unsigned
            // 排列是：{row0 的 k, k+1}, {row1 的 k, k+1}, {row0 的 k+8, k+9}, {row1 的 k+8, k+9}
            // 而 PV 的 k 段有 16 个 key = 2 个 n-tile（每个 n-tile 8 个 key）

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
        }   // if (jt <= jt_end_wg)

        __syncthreads();   // 所有 warpgroup 都算完，才能复用 buf^1
        buf ^= 1;
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
// [HOST]
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
    fa2_wgmma_kernel<<<grid, SM_THREADS>>>(
        (const half*)q.data_ptr(), (const half*)k.data_ptr(), (const half*)v.data_ptr(),
        (float*)o.data_ptr(), B, H, Hkv, S, (float)scale);
    const cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "fa2_wgmma launch failed: ", cudaGetErrorString(err));
    return o;
}
