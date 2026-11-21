/*
    注：
        1、wmma 是 Ampere 之前的 CUDA C++ API（nvcuda::wmma::fragment），w12b 用的是这个
        2、mma.sync 是 PTX 指令（warp 级、同步），w12c/w12d/w12e 的 PV 用的是这个
        3、wgmma 是 Hopper 的 warpgroup 级、异步指令，w12e 的 QK 用的是这个


    
    S = Q @ K^T  通过 wgmma 算，数据通过 TMA 搬运 + 双缓冲  ———— Q、K的描述符各自独立，一个 core-matrix 一个 swizzle 是合法的
    O = P @ V  还是 mma.sync 算

    wgmma的矩阵描述符不是 必须 core-matrix，而是 描述符必须和物理布局对上
    Q 矩阵没有使用 TMA 或 cp.async，而是由 256 个线程通过普通的同步 store 指令手动拷贝并重排进 SMEM 的
    Q的描述符是 core-matrix，K是 swizzle 

    硬件只认两组布局词汇，描述符的 swizzle 位就是选词：

    swizzle	        硬件假设的物理布局	                                                    谁产的出来
    0	            每个 8×8 块 = 连续 128 B（8 行压在一起），块间靠 LBO/SBO 跳	        线程、cp.async（能算任意地址）
    1 / 2 / 3	    行主序，行宽 128 / 64 / 32 B，行内 16 B 块按行号异或	            TMA（直出）、线程、cp.async


*/


#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include <cstdio>


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
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
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


// 这三行都没用了，都用的 TMA，没有走 cp.async了
__device__ __forceinline__ void cp16(half* dst, const half* src, bool ok)
{
    const unsigned s = (unsigned)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(src),
                 "r"(ok ? 16 : 0));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
__device__ __forceinline__ void cp_wait1() { asm volatile("cp.async.wait_group 1;\n" ::: "memory"); }


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

/*
    K/V 是 3维的 [B*Hkv][S][D]
    所以 TMA 搬运也要是 3D 的
    gpt2 的 KV 张量是 [B, Hkv, S, D]，展平后就是三层：
        维度 2（最外）：bh = b*Hkv + kvh  ————> 一共 B*Hkv 个"头"
        维度 1（中间）：token 位置 0..S-1
        维度 0（最内）：head_dim 0..63（每个 128 B = 64 个 half）
    
    block 要读的是 其中一个头的一段token，也就是这个 3D 张量的一个 {64, 32, 1} 小方块
        box = {D=64, TILE_N=32, 1}
        坐标 = (0, jt*TILE_N, bh)， 内层在前，正好对应 tma_load_3d(Kbuf, bar, map, 0, jt*TILE_N, bh)
*/
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
__global__ void __launch_bounds__(SM_THREADS) fa2_wgmma_kernel(const __grid_constant__ CUtensorMap vmap,
                                                                const __grid_constant__ CUtensorMap kmap,
                                                                const half* __restrict__ Q, 
                                                                const half* __restrict__ K,         // K/V都靠描述符，这两个参数没用了
                                                                const half* __restrict__ V,
                                                                float* __restrict__ O, 
                                                                int B, int H, int Hkv, int S, float scale)
{
    __shared__ alignas(128) half Qs[TILE_M * HEAD_DIM];         // 16 KB，core-matrix
    __shared__ alignas(128) half Kbuf[2][TILE_N * HEAD_DIM];    // 2 × 4 KB，128B swizzle（TMA 直出）
    __shared__ alignas(128) half Vbuf[2][TILE_N * HEAD_DIM];    // 2 × 4 KB，swizzle

    __shared__ alignas(8) uint64_t vbar[2];         // K/V TMA时 共用的mbar

    const int bx = blockIdx.x;
    const int bh = blockIdx.y;
    const int b = bh / H;
    const int h = bh % H;
    const int kvh = h / (H / Hkv);
    const int kv_bh = b * Hkv + kvh;        // 描述符第三维的下标

    const int row_base = max(0, min(bx * TILE_M, S - TILE_M));

    const size_t q_base = ((size_t)b * H + h) * S * HEAD_DIM;

    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int warp_id = tid / 32;       // 0..7
    const int wg = warp_id / 4;         // 第几个 warpgroup：0/1
    const int wi = warp_id % 4;         // warpgroup 内第几个 warp：0..3

    // 因为 mma.sync/wgmma 的 C 布局，所以是4个lane管tile的一行
    // 本线程负责的两行（tile 内下标）
    const int row0 = wg * WWG + wi * WARP_M + (lane / 4);           //  lane / 4	0..7	warp 内第几行（4 条 lane 一组管一行）
    const int row1 = row0 + 8;                                      // 同一个lane   +8 行

    // 初始化两个 mbarrier
    if(tid == 0)
    {
        mbar_init(&vbar[0], 1);
        mbar_init(&vbar[1], 1);
    }
    __syncthreads();

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
        // K、V 都走 TMA：一条 mbarrier 同时等两份数据（4096 + 4096 = 8192 B）
        if (tid == 0)
        {
            mbar_arrive_expect_tx(&vbar[buf], 2 * TILE_N * HEAD_DIM * (uint32_t)sizeof(half));
            tma_load_3d(Kbuf[buf], &vbar[buf], &kmap, 0, jtile * TILE_N, kv_bh);    // 这里要传的是 kv_bh 不是 bh
            tma_load_3d(Vbuf[buf], &vbar[buf], &vmap, 0, jtile * TILE_N, kv_bh);
        }
    };

    // ---- online softmax 状态，还是在寄存器里 ----
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

    int buf = 0;
    for (int jt = 0; jt <= jt_end; ++jt)
    {
        if (jt + 1 <= jt_end)
            load_kv(jt + 1, buf ^ 1);               // 先把下一个 tile 发出去
        mbar_wait(&vbar[buf], (jt >> 1) & 1);       // 等数据搬完；该指令是每个线程各自执行的；整条 barrier 的字节全到齐，不需要再__syncthreads()       


        if (jt <= jt_end_wg)
        {
            // ---- S = Q·Kᵀ：4 个 k 段 ----
            float sc[TILE_N / 8][4];
            for (int n = 0; n < TILE_N / 8; ++n)
                for (int c = 0; c < 4; ++c) sc[n][c] = 0.f;

            const unsigned a_lbo = 128, a_sbo = 1024;     // A(Q)：core-matrix，swizzle=0
            const unsigned b_lbo = 1,   b_sbo = 1024;     // B(K)：128B swizzle，LBO 不用、SBO = 8 行 × 128 B

            half* qwg = Qs + wg * (WWG * HEAD_DIM);   // 本 warpgroup 的 64 行
            half* kt = Kbuf[buf];

            wgmma_fence();                            // 保证 smem 已就绪
            for (int kk = 0; kk < HEAD_DIM / 16; ++kk)
            {
                const unsigned long long ad = make_desc(qwg + kk * 128, a_lbo, a_sbo, 0);
                // K 的 128B swizzle 布局里，第 kk 个 16-k 段是 行内第 kk 个 32 B，不是 128 B 的整数倍
                // 这里 K 的描述符 swizzle = 1，描述符里的 swizzle 位必须和 smem 里的物理布局一致
                const unsigned long long bd = make_desc(kt + kk * 16, b_lbo, b_sbo, 1);
                wgmma_m64n32k16(sc, ad, bd);            // wgmma
            }
            wgmma_commit();
            wgmma_wait0();          // 阻塞等待所有组完成


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
                    const half* p = Vbuf[buf] + swz(vr, nd * 8);       // V 的 swizzle 布局是 TMA 按描述符写出来的
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

// 构建 3D TMA 描述符
static int make_tensor_map(CUtensorMap *tmap, void *ptr, int BH, int S, int D)
{
    // K/V 形状 [B*Hkv][S][D]
    // 在TMA里面x是连续维，y是跳跃维
    cuuint64_t gdim[3]    = {(cuuint64_t)D, (cuuint64_t)S, (cuuint64_t)BH};     // 每个维度的元素个数
    cuuint64_t gstride[2] = {(cuuint64_t)D * sizeof(half),                      // 行跨度 = 128 B
                             (cuuint64_t)D * sizeof(half) * S};                 // head 跨度
    cuuint32_t box[3]     = {(cuuint32_t)D, (cuuint32_t)TILE_N, 1};             // 每次搬多少个元素
    cuuint32_t estride[3] = {1, 1, 1};  

    CUresult r = cuTensorMapEncodeTiled(
        tmap, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 3, 
        ptr, gdim, gstride,
        box, estride, 
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,                    // smem 里读K/V用的是128B swizzle
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

    fa2_wgmma_kernel<<<grid, SM_THREADS>>>(
        vmap, kmap,
        (const half*)q.data_ptr(), (const half*)k.data_ptr(), (const half*)v.data_ptr(),
        (float*)o.data_ptr(), B, H, Hkv, S, (float)scale);
    const cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "fa2_wgmma launch failed: ", cudaGetErrorString(err));
    return o;
}
