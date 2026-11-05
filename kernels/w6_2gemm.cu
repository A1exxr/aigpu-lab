// ============================================================================
// W6_1gemm.cu —— GEMM 台阶5：向量化装载 + cp.async 双缓冲
//
// 基准：cuBLAS fp16（同一进程、同一 event、交替计时）—— 本文件的唯一比较基准
// 历史：台阶4（wmma）0.2836 ms → 5b 0.1754 ms → 6 0.1232 ms（cuBLAS 0.0811 ms）
//
// 台阶5 拆成两个子步骤，一次只改一件事：
//   5a：把装载从「逐元素 half」改成「uint4 一次 16 B」  ← 本文件先做这个
//   5b：把装载换成 cp.async + 双缓冲（等 5a 测完再加）
//
// 编译 & 运行：
//   nvcc -O3 -arch=sm_89 W6_1gemm.cu -o W6_1gemm && ./W6_1gemm
// ============================================================================

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

using namespace nvcuda;

// 注意：这里用 constexpr 而不是 #define。
// #define M 1024 会让「int M」这种参数名在预处理阶段被替换成「int 1024」，
// 报错是 `expected a ")"` + `too many arguments in function call`，很难看出原因。
constexpr int M = 1024;
constexpr int N = 1024;
constexpr int K = 1024;

// ----------------------------------------------------------------------------
// 台阶 5a：装载向量化（uint4 = 16 B = 8 个 half）
//
// 关键：一个 32x32 的 half tile 一行是 32 个 half = 64 B = 4 个 uint4
//       32 行 x 4 个 uint4 = 128 个 uint4，而我们有 128 个线程
//       ——> 每个线程刚好搬「一个 uint4」，一共 1 条 LDG.128 + 1 条 STS.128
//
// 对照：原来的写法每线程搬 8 个 half，是 8 条 LDG.16 + 8 条 STS.16
//       指令数 16 条 -> 2 条，缩小 8 倍
// ----------------------------------------------------------------------------

// uint4 是 CUDA 内置的 16 字节向量类型（4 个 unsigned int），和 float4/int4 大小、对齐完全一样
// 它的唯一目的是让一条指令搬 16 字节 ———— 减少指令数、合并内存事务、拉满 HBM 带宽，是访存密集型 kernel（GEMM、cp.async 搬运）的标准手段
// 用 uint4 而非 float4 来搬原始数据，是因为搬运不关心内容类型，uint4 的语义更中性
// 代价是必须 16 字节对齐（i 从 4 的倍数开始、步长也是 4 的倍数），否则 misaligned 或降速

__global__ void gemm_tc5a_kernel(const half* __restrict__ A,
                                 const half* __restrict__ B,
                                 half* __restrict__ C,
                                 int m, int k, int n)
{
    int tx   = threadIdx.x;
    int ty   = threadIdx.y;
    int btid = ty * 32 + tx;

    int warp_m = ty / 2;
    int warp_n = ty % 2;
    int bm = blockIdx.y;
    int bn = blockIdx.x;

    __shared__ half  As[32][32];
    __shared__ half  Bs[32][32];
    __shared__ float Cf[32][32];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (int k0 = 0; k0 < k; k0 += 32)
    {
        // 每个线程搬一个 uint4，不需要循环，一个uint4是16字节，就是8个half，128个线程每个线程直接搬8个元素
        // 就不需要for(int i = btid; i < 1024; i += blockDim.x)了，要跑 8 轮（每轮 2 字节）
        // block是(32,4)  ty 0 1 2 3； tx是0 1 ... 31
        // 一行有4个线程，每个线程都搬了8个元素，就是一行32个元素， 32*32=1024，刚好是一个C块
        // As就是A的一个32*32子块的拷贝，所以子块内的相对坐标(r,cvec)能同时定位两边
        /*
            坐标	    在 As（shared）里	            在 A（global）里
            r	        第 r 行（0..31）	            第 bm*32 + r 行（全局行）
            cvec*8	    第 cvec*8 列（0,8,16,24）	    第 k0 + cvec*8 列（全局列）
        */ 

        int r = btid / 4;           // 块内第几行   (0..31)，每4个线程换一行
        int cvec = btid % 4;        // 行内第几个uint4  (0 1 2 3)

        // 加载As
        // reinterpret_cast 源和目标的地址都要
        // 源 A 是 const half* → const uint4*
        // 目的 As 是非 const → uint4*
        // 指令数缩减了8倍
        // 告诉编译器"按 16 字节搬" ———— 类型从 half* 变成 uint4*，编译器生成 LDG.128/STS.128
        // reinterpret_cast 是无检查的强制转换，但必须手动保证（基址、行宽、块偏移、块内偏移全是 16 的倍数）
        const uint4* srcA = reinterpret_cast<const uint4*>(&A[(bm * 32 + r) * k + k0 + cvec * 8]);  
        uint4* dstA = reinterpret_cast<uint4*>(&As[r][cvec * 8]);
        *dstA = *srcA;

        // 加载Bs
        const uint4* srcB = reinterpret_cast<const uint4*>(&B[(k0 + r) * n + bn * 32 + cvec * 8]);
        uint4* dstB = reinterpret_cast<uint4*>(&Bs[r][cvec * 8]);
        *dstB = *srcB;
        __syncthreads();

        for (int kk = 0; kk < 32; kk += 16)
        {
            wmma::load_matrix_sync(a_frag, &As[warp_m * 16][kk], 32);
            wmma::load_matrix_sync(b_frag, &Bs[kk][warp_n * 16], 32);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        __syncthreads();
    }

    wmma::store_matrix_sync(&Cf[warp_m * 16][warp_n * 16], c_frag, 32, wmma::mem_row_major);
    __syncthreads();

    for (int i = btid; i < 1024; i += 128)
    {
        int r = i / 32;
        int c = i % 32;
        C[(bm * 32 + r) * n + bn * 32 + c] = __float2half(Cf[r][c]);
    }
}
// ----------------------------------------------------------------------------
// 台阶 5b：cp.async + 双缓冲
//
// 与 5a 的唯一区别：装载不再用普通 load/store，而是
//   __pipeline_memcpy_async(dst_smem, src_global, 16)     // 发一个异步拷贝，立刻返回，数据在路上 ———— 直接 global -> shared，不过寄存器
//   __pipeline_commit()                                   // 把"自上次 commit 以来发的所有 async copy"打成一个组 ———— 提交这一批拷贝
//   __pipeline_wait_prior(1)                              // 等到"最多还剩 N 个组未完成"； n取1就是 等到“只剩 1 批没完成”
//      wait_prior(1) 是"按 commit 顺序保留最新 1 组

//   memcpy_async 是"发出去就返回"，不等数据到；wait_prior 才是真正的等待点，没有 wait_prior，读到的是还没搬完的垃圾
// 并且用两块共享内存轮流接，让“搬下一块”和“算当前块”重叠。
//
// 现在这个 kernel 还是单缓冲 + 同步版本（cur 恒为 0），
// 跑出来的数应该和 5a 完全一样。你要做的是把它改成双缓冲异步。
// ----------------------------------------------------------------------------
#include <cuda_pipeline.h>

__global__ void gemm_tc5b_kernel(const half* __restrict__ A,
                                 const half* __restrict__ B,
                                 half* __restrict__ C,
                                 int m, int k, int n)
{
    int tx   = threadIdx.x;
    int ty   = threadIdx.y;
    int btid = ty * 32 + tx;

    int warp_m = ty / 2;
    int warp_n = ty % 2;
    int bm = blockIdx.y;
    int bn = blockIdx.x;

    // 双缓冲，每块共享内存要 16 B 对齐
    __shared__ __align__(16) half As[2][32][32];
    __shared__ __align__(16) half Bs[2][32][32];
    __shared__ float Cf[32][32];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    int r    = btid / 4;      // 第几行
    int cvec = btid % 4;      // 行内第几个 uint4

    // ---- 异步搬第 0 块到 buf 0；只 commit，不 wait ----
    __pipeline_memcpy_async(&As[0][r][cvec * 8], &A[(bm * 32 + r) * k + cvec * 8], 16);
    __pipeline_memcpy_async(&Bs[0][r][cvec * 8], &B[r * n + bn * 32 + cvec * 8], 16);
    __pipeline_commit();        // 这里只需要打包成一个组，如果这里也wait，那就退化成同步了

    int cur = 0;
    for (int k0 = 0; k0 < k; k0 += 32)
    {
        int nxt = 1 - cur;
        bool has_next = (k0 + 32 < k);

        // 预取下一块，到另一块缓冲
        if (has_next)
        {
            __pipeline_memcpy_async(&As[nxt][r][cvec * 8],
                                    &A[(bm * 32 + r) * k + (k0 + 32) + cvec * 8], 16);
            __pipeline_memcpy_async(&Bs[nxt][r][cvec * 8],
                                    &B[(k0 + 32 + r) * n + bn * 32 + cvec * 8], 16);
            __pipeline_commit();
        }

        // 等到 最多还剩 1 批没完成 = 当前这块已落地
        // 最后一块没有预取，只剩 1 批，所以用 0 全等
        __pipeline_wait_prior(has_next ? 1 : 0);
        __syncthreads();

        // 算当前块
        for (int kk = 0; kk < 32; kk += 16)
        {
            wmma::load_matrix_sync(a_frag, &As[cur][warp_m * 16][kk], 32);
            wmma::load_matrix_sync(b_frag, &Bs[cur][kk][warp_n * 16], 32);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        __syncthreads();

        // 翻缓冲
        cur = nxt;
    }

    wmma::store_matrix_sync(&Cf[warp_m * 16][warp_n * 16], c_frag, 32, wmma::mem_row_major);
    __syncthreads();

    for (int i = btid; i < 1024; i += 128)
    {
        int rr = i / 32;
        int cc = i % 32;
        C[(bm * 32 + rr) * n + bn * 32 + cc] = __float2half(Cf[rr][cc]);
    }
}

// ----------------------------------------------------------------------------
// 台阶 5c：三级流水（As[3] / Bs[3]）
//
// 5b 是两级：算当前、搬下一块 —— 只有 1 块在路上。
// 5c 是三级：算当前、搬下一块、再下一块也已经发出去了 —— 2 块在路上。
//
// 唯一的难点：__pipeline_wait_prior(N) 里的 N 必须等于「此刻还在飞、但已经用不到的批次数」。
// 循环里最多是 2（i+1、i+2 两块），但最后两轮没有那么多可预取，N 要退化成 1、0。
//   写成固定 2 -> 最后几轮会等到“永久不属于任何块”的批次，卡死或读错
//   写成固定 0 -> 每轮全等，退化回同步
// ----------------------------------------------------------------------------
__global__ void gemm_tc5c_kernel(const half* __restrict__ A,
                                 const half* __restrict__ B,
                                 half* __restrict__ C,
                                 int m, int k, int n)
{
    int tx   = threadIdx.x;
    int ty   = threadIdx.y;
    int btid = ty * 32 + tx;

    int warp_m = ty / 2;
    int warp_n = ty % 2;
    int bm = blockIdx.y;
    int bn = blockIdx.x;

    __shared__ __align__(16) half As[3][32][32];
    __shared__ __align__(16) half Bs[3][32][32];
    __shared__ float Cf[32][32];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    int r    = btid / 4;
    int cvec = btid % 4;

    // 一共几个 k 块
    const int NB = k / 32;

/*
    定义一个宏，宏是文本替换，不是函数调用
    blk 是"K 维的第几个 32 列块"
    把第 blk 块异步搬进缓冲 buf，并单独 commit（= 一批）

    blk 和 buf 是模板参数，ISSUE(0, 0)，替换时把 0、0 填进去
    编译器看到的就是下面三行，把0和0带进去：
        // ISSUE(0, 0) 展开后：
        __pipeline_memcpy_async(&As[0][r][cvec * 8],
                                &A[(bm * 32 + r) * k + 0 * 32 + cvec * 8], 16);
        __pipeline_memcpy_async(&Bs[0][r][cvec * 8],
                                &B[(0 * 32 + r) * n + bn * 32 + cvec * 8], 16);
        __pipeline_commit();
*/
#define ISSUE(blk, buf)                                                              \
    __pipeline_memcpy_async(&As[buf][r][cvec * 8],                                   \
                            &A[(bm * 32 + r) * k + (blk) * 32 + cvec * 8], 16);      \
    __pipeline_memcpy_async(&Bs[buf][r][cvec * 8],                                   \
                            &B[((blk) * 32 + r) * n + bn * 32 + cvec * 8], 16);      \
    __pipeline_commit();        // 把这 256 个 copy 打包成 1 组
    // 宏的定义到这里就结束了，  \是续行符，含义是"这一行还没完，下一行还是同一个逻辑行"——最后一行已经完了，不需要再"续"
    // 结束的这一行就不用加这个续行符了

    // 第 0、1 块先发出去（真正搬的时候不占计算时间）
    // 三级流水的目标是"算 i 时，i+1 和 i+2 都在飞"
    ISSUE(0, 0)             // 搬 K 维第 0 块（k0=0）  到 buf 0
    if (NB > 1) {
        ISSUE(1, 1)         // 搬 K 维第 1 块（k0=32） 到 buf 1
    }

    for (int i = 0; i < NB; ++i)
    {
        // 本轮算哪个 buf
        int cur = i % 3;    

        // 预取 i+2 那一块，放到 (i+2)%3 号缓冲
        if (i + 2 < NB) {
            ISSUE(i + 2, (i + 2) % 3)           // (i+2)%3 = (i-1)%3， 是上一轮（i-1）用过的 buffer
        }

        // 还在飞、但用不到的批次：最多 2 个（i+1、i+2）；末尾会变少
        int remain = NB - 1 - i;                        // 当前 i 之后还剩多少块
        int wait_n = remain < 2 ? remain : 2;           // 最多允许 2 组在飞
        __pipeline_wait_prior(wait_n);
        __syncthreads();

        for (int kk = 0; kk < 32; kk += 16)
        {
            wmma::load_matrix_sync(a_frag, &As[cur][warp_m * 16][kk], 32);
            wmma::load_matrix_sync(b_frag, &Bs[cur][kk][warp_n * 16], 32);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        __syncthreads();
    }
// 取消宏定义
#undef ISSUE

    wmma::store_matrix_sync(&Cf[warp_m * 16][warp_n * 16], c_frag, 32, wmma::mem_row_major);
    __syncthreads();

    for (int idx2 = btid; idx2 < 1024; idx2 += 128)
    {
        int rr = idx2 / 32;
        int cc = idx2 % 32;
        C[(bm * 32 + rr) * n + bn * 32 + cc] = __float2half(Cf[rr][cc]);
    }
}

// ----------------------------------------------------------------------------
// 台阶 6：C tile 64×64，每 warp 算 32×32（4 个 16×16 fragment）
//
// 与台阶5b（C tile 32×32，每 warp 16×16）的差别只有两处：
//   ① tile 大 4 倍 → K 循环里每装一块 A/B，能算的 C 元素多 4 倍（smem 复用率翻倍）
//   ② 每个 warp 持有 4 个累加器 → 4 次 mma 摊 2 次 load（a/b 各 2 次）
//
// smem：As[2][64][32] + Bs[2][32][64] = 16 KB（双缓冲）
// 线程：block(32,4) = 128 = 4 warp，warp 布局 2×2，每 warp 管 32×32
// 直写全局，不再用 Cf 中转（省 16 KB，否则只有 3 个 block 能驻留）
// ----------------------------------------------------------------------------
#define TILE6 64

__global__ void gemm_tc6_kernel(const half* __restrict__ A,
                                const half* __restrict__ B,
                                half* __restrict__ C,
                                int m, int k, int n)
{
    int tx   = threadIdx.x;
    int ty   = threadIdx.y;
    int btid = ty * 32 + tx;

    int warp_m = ty / 2;      // 0..1
    int warp_n = ty % 2;      // 0..1
    int bm = blockIdx.y;      // 行块号（M 方向）
    int bn = blockIdx.x;      // 列块号（N 方向）

    __shared__ __align__(16) half As[2][TILE6][32];
    __shared__ __align__(16) half Bs[2][32][TILE6];
    // float 累加器只能 store 到 float*，所以需要一个 float 中转区（16 KB）
    __shared__ __align__(16) float Cf[TILE6][TILE6];

    // 每个 warp：a 两片、b 两片、c 四个
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag[2];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag[2];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[4];
    for (int i = 0; i < 4; ++i) {
        wmma::fill_fragment(c_frag[i], 0.0f);
    }

    const int NB = k / 32;    // k 方向的块数

// 把第 blk 个 k 块搬进缓冲 buf。每线程每次搬 8 个 half = 16 B，
// A tile 有 64×32/8 = 256 块 → 128 线程搬 2 轮；B tile 32×64 同理。
#define LOAD6(blk, buf)                                                             \
    for (int t = 0; t < 2; ++t) {                                                   \
        int ch = t * 128 + btid;                                                    \
        int ra = ch / 4, ca = (ch % 4) * 8;                                         \
        __pipeline_memcpy_async(&As[buf][ra][ca],                                   \
                                &A[(bm * TILE6 + ra) * k + (blk) * 32 + ca], 16);   \
        int rb = ch / 8, cb = (ch % 8) * 8;                                         \
        __pipeline_memcpy_async(&Bs[buf][rb][cb],                                   \
                                &B[((blk) * 32 + rb) * n + bn * TILE6 + cb], 16);   \
    }                                                                               \
    __pipeline_commit();

    LOAD6(0, 0)

    for (int i = 0; i < NB; ++i)
    {
        int cur = i % 2;

        if (i + 1 < NB) {
            LOAD6(i + 1, (i + 1) % 2)
        }

        int remain = NB - 1 - i;
        __pipeline_wait_prior(remain < 1 ? 0 : 1);
        __syncthreads();

        for (int kk = 0; kk < 32; kk += 16)
        {
            // 一次装 2 片 a、2 片 b，然后做 4 次 mma（这就是复用率的来源）
            wmma::load_matrix_sync(a_frag[0], &As[cur][warp_m * 32 + 0][kk], 32);
            wmma::load_matrix_sync(a_frag[1], &As[cur][warp_m * 32 + 16][kk], 32);
            wmma::load_matrix_sync(b_frag[0], &Bs[cur][kk][warp_n * 32 + 0], TILE6);
            wmma::load_matrix_sync(b_frag[1], &Bs[cur][kk][warp_n * 32 + 16], TILE6);
            for (int ii = 0; ii < 2; ++ii) {
                for (int jj = 0; jj < 2; ++jj) {
                    wmma::mma_sync(c_frag[ii * 2 + jj], a_frag[ii], b_frag[jj], c_frag[ii * 2 + jj]);
                }
            }
        }
        __syncthreads();
    }
#undef LOAD6

    // 先落到 smem 的 Cf（float），再统一转 half 写回全局
    for (int ii = 0; ii < 2; ++ii) {
        for (int jj = 0; jj < 2; ++jj) {
            float* pdst = &Cf[warp_m * 32 + ii * 16][warp_n * 32 + jj * 16];
            wmma::store_matrix_sync(pdst, c_frag[ii * 2 + jj], TILE6, wmma::mem_row_major);
        }
    }
    __syncthreads();

    for (int idx = btid; idx < TILE6 * TILE6; idx += 128) {
        int rr = idx / TILE6;
        int cc = idx % TILE6;
        C[(size_t)(bm * TILE6 + rr) * n + bn * TILE6 + cc] = __float2half(Cf[rr][cc]);
    }
}

// ----------------------------------------------------------------------------
// host
// ----------------------------------------------------------------------------
#define CHECK(x)                                                                          \
    do {                                                                                  \
        cudaError_t e = (x);                                                              \
        if (e != cudaSuccess) {                                                           \
            printf("CUDA error %s at line %d: %s\n", #x, __LINE__, cudaGetErrorString(e)); \
            exit(1);                                                                      \
        }                                                                                 \
    } while (0)

__global__ void fill_kernel(half* p, int n, int mode)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        p[i] = (mode == 0) ? __float2half((float)(i % 37) * 0.01f - 0.18f)
                           : __float2half((float)(i % 53) * 0.013f - 0.34f);
    }
}

// ----------------------------------------------------------------------------
// cuBLAS 对照：同一进程、同一 event、同一份数据，交替计时
//
// 注意 cuBLAS 是列主序。A 行主序 M×K == 列主序 K×M（lda=K）；
// B 行主序 K×N == 列主序 N×K（lda=N）；于是
//     C_cm(N×M) = B_cm(N×K) · A_cm(K×M)
// 即调用时 m=N, n=M, k=K，两个矩阵指针顺序对调。
// 本文件的比较基准只有 cuBLAS；台阶4 在 w6_gemm.cu 里，不参与本文件任何比较。
// ----------------------------------------------------------------------------
static cublasHandle_t g_handle = nullptr;

void launch_cublas(const half* A, const half* B, half* C)
{
    float alpha = 1.f, beta = 0.f;
    cublasGemmEx(g_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                 N, M, K,
                 &alpha,
                 B, CUDA_R_16F, N,      // B_cm: N×K, lda = N
                 A, CUDA_R_16F, K,      // A_cm: K×M, ldb = K
                 &beta,
                 C, CUDA_R_16F, N,      // C_cm: N×M, ldc = N
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

float bench(void (*launch)(const half*, const half*, half*), const half* dA,
            const half* dB, half* dC, int warmup, int rounds, int iters)
{
    for (int i = 0; i < warmup; ++i) launch(dA, dB, dC);
    CHECK(cudaDeviceSynchronize());

    float best = 1e30f;
    for (int rr = 0; rr < rounds; ++rr) {
        cudaEvent_t s, e;
        cudaEventCreate(&s);
        cudaEventCreate(&e);
        cudaEventRecord(s);
        for (int i = 0; i < iters; ++i) launch(dA, dB, dC);
        cudaEventRecord(e);
        cudaEventSynchronize(e);
        float ms = 0;
        cudaEventElapsedTime(&ms, s, e);
        if (ms / iters < best) best = ms / iters;
        cudaEventDestroy(s);
        cudaEventDestroy(e);
    }
    return best;
}

static dim3 BLOCK(32, 4);
static dim3 GRID(N / 32, M / 32);

void launch_tc5a(const half* A, const half* B, half* C)
{
    gemm_tc5a_kernel<<<GRID, BLOCK>>>(A, B, C, M, K, N);
}

void launch_tc5b(const half* A, const half* B, half* C)
{
    gemm_tc5b_kernel<<<GRID, BLOCK>>>(A, B, C, M, K, N);
}
void launch_tc5c(const half* A, const half* B, half* C)
{
    gemm_tc5c_kernel<<<GRID, BLOCK>>>(A, B, C, M, K, N);
}
// 台阶6 的 grid 是 64×64 一块，所以比 GRID 少一半
void launch_tc6(const half* A, const half* B, half* C)
{
    dim3 grid6(N / TILE6, M / TILE6);
    gemm_tc6_kernel<<<grid6, BLOCK>>>(A, B, C, M, K, N);
}

int main()
{
    size_t bytesA = (size_t)M * K * sizeof(half);
    size_t bytesB = (size_t)K * N * sizeof(half);
    size_t bytesC = (size_t)M * N * sizeof(half);

    half *dA, *dB, *dC0, *dC1, *dC2;
    CHECK(cudaMalloc(&dA, bytesA));
    CHECK(cudaMalloc(&dB, bytesB));
    CHECK(cudaMalloc(&dC0, bytesC));
    CHECK(cudaMalloc(&dC1, bytesC));
    CHECK(cudaMalloc(&dC2, bytesC));
    cublasCreate(&g_handle);

    int threads = 256;
    fill_kernel<<<(M * K + threads - 1) / threads, threads>>>(dA, M * K, 0);
    fill_kernel<<<(K * N + threads - 1) / threads, threads>>>(dB, K * N, 1);
    CHECK(cudaDeviceSynchronize());

    // 正确性：唯一参照是 cuBLAS（fp16 输出，容差 1e-2）
    launch_cublas(dA, dB, dC2);
    CHECK(cudaDeviceSynchronize());

    void (*cand[4])(const half*, const half*, half*) = {launch_tc5a, launch_tc5b, launch_tc5c, launch_tc6};
    half* bufs[4] = {dC0, dC0, dC1, dC1};
    const char* names[4] = {"5a uint4", "5b cp.async x2", "5c cp.async x3", "6  64x64 tile"};

    half* hcu = (half*)malloc(bytesC);
    half* hout = (half*)malloc(bytesC);
    CHECK(cudaMemcpy(hcu, dC2, bytesC, cudaMemcpyDeviceToHost));

    printf("\n=== 正确性（对 cuBLAS，fp16 输出，1e-2） ===\n");
    for (int i = 0; i < 4; ++i) {
        cand[i](dA, dB, bufs[i]);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(hout, bufs[i], bytesC, cudaMemcpyDeviceToHost));
        float mx = 0.f;
        for (int j = 0; j < M * N; ++j) {
            float d = fabsf(__half2float(hout[j]) - __half2float(hcu[j]));
            if (d > mx) mx = d;
        }
        printf("max |%-16s - cuBLAS| = %.3e  %s\n", names[i], mx, mx < 1e-2f ? "[OK]" : "[WRONG]");
    }

    const double FLOPS = 2.0 * M * N * K;
    const double PEAK = 65.2;   // RTX 4060 Laptop fp16 Tensor Core 峰值 TFLOPS（实测 2655 MHz）
    float best_cu = 1e30f, best[4] = {1e30f, 1e30f, 1e30f, 1e30f};

    // 交替跑：同一时刻的频率/温度状态，避免“谁先跑谁吃亏”
    for (int round = 0; round < 5; ++round) {
        float cu = bench(launch_cublas, dA, dB, dC2, 100, 1, 100);
        if (cu < best_cu) best_cu = cu;
        for (int i = 0; i < 4; ++i) {
            float v = bench(cand[i], dA, dB, bufs[i], 100, 1, 100);
            if (v < best[i]) best[i] = v;
        }
    }

    printf("\n=== 性能（M=N=K=1024, fp16；min-of-5 轮 x 100 次） ===\n");
    printf("%-16s %8s %8s %7s %8s\n", "实现", "ms", "TFLOPS", "%峰值", "%cuBLAS");
    double tf_cu = FLOPS / (best_cu * 1e-3) / 1e12;
    printf("%-16s %8.4f %8.2f %6.1f%% %7.1f%%\n", "cuBLAS fp16", best_cu, tf_cu,
           tf_cu / PEAK * 100.0, 100.0);
    for (int i = 0; i < 4; ++i) {
        double tf = FLOPS / (best[i] * 1e-3) / 1e12;
        printf("%-16s %8.4f %8.2f %6.1f%% %7.1f%%\n", names[i], best[i], tf,
               tf / PEAK * 100.0, best_cu / best[i] * 100.0);
    }
    printf("\n注：%%cuBLAS = 我方达到 cuBLAS 的比例；1.52x 慢 ≈ 66%%\n");

    cublasDestroy(g_handle);
    return 0;
}
