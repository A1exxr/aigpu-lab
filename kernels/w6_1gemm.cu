// ============================================================================
// W6_1gemm.cu —— GEMM 台阶5：向量化装载 + cp.async 双缓冲
//
// 基线：台阶4（wmma，block 32x4，C tile 32x32）实测 0.2836 ms / 7.57 TFLOPS
// 目标：0.20 ~ 0.23 ms
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
// 基线：台阶4 原样搬过来（已验证正确，0.2836 ms）
// ----------------------------------------------------------------------------
__global__ void gemm_tc_kernel(const half* __restrict__ A,
                               const half* __restrict__ B,
                               half* __restrict__ C,
                               int m, int k, int n)
{
    int tx   = threadIdx.x;               // 0..31   tile 内的列偏移
    int ty   = threadIdx.y;               // 0..3    tile 内的 warp 编号
    int btid = ty * 32 + tx;              // 0..127  一个block 128个线程，要处理32*32=1024个数据，一个线程要搬8个数据

    int warp_m = ty / 2;                  // 0 1
    int warp_n = ty % 2;                  // 0 1
    int bm = blockIdx.y;                  // 行方向：第几个 32 行的 tile
    int bn = blockIdx.x;                  // 列方向：第几个 32 列的 tile

    __shared__ half  As[32][32];
    __shared__ half  Bs[32][32];
    __shared__ float Cf[32][32];          // WMMA 的累加器用float类型

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (int k0 = 0; k0 < k; k0 += 32)
    {
        // 128线程装1024个元素，1个线程装8个元素
        for (int i = btid; i < 1024; i += 128)
        {
            int r = i / 32;
            int c = i % 32;
            As[r][c] = A[(bm * 32 + r) * k + (k0 + c)];
            Bs[r][c] = B[(k0 + r) * n + (bn * 32 + c)];
        }
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
// 台阶 5a：装载向量化（uint4 = 16 B = 8 个 half）
//
// 关键：一个 32x32 的 half tile 一行是 32 个 half = 64 B = 4 个 uint4。
//       32 行 x 4 个 uint4 = 128 个 uint4，而我们有 128 个线程
//       ——> 每个线程刚好搬「一个 uint4」，一共 1 条 LDG.128 + 1 条 STS.128。
//
// 对照：原来的写法每线程搬 8 个 half，是 8 条 LDG.16 + 8 条 STS.16。
//       指令数 16 条 -> 2 条，缩小 8 倍。
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
// host：填充 -> 校验（5a 必须和基线逐位接近）-> 计时（min-of-5 轮）
// ----------------------------------------------------------------------------
#define CHECK(x)                                                                      \
    do {                                                                              \
        cudaError_t e = (x);                                                          \
        if (e != cudaSuccess) {                                                       \
            printf("CUDA error %s at line %d: %s\n", #x, __LINE__, cudaGetErrorString(e)); \
            exit(1);                                                                  \
        }                                                                             \
    } while (0)

__global__ void fill_kernel(half* p, int n, int mode)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // 数据必须有变化，否则「读错行」这类 bug 会恰好算出正确答案
        p[i] = (mode == 0) ? __float2half((float)(i % 37) * 0.01f - 0.18f)
                           : __float2half((float)(i % 53) * 0.013f - 0.34f);
    }
}

float bench(void (*launch)(const half*, const half*, half*), const half* dA,
            const half* dB, half* dC, int warmup, int rounds, int iters)
{
    for (int i = 0; i < warmup; ++i) launch(dA, dB, dC);
    CHECK(cudaDeviceSynchronize());

    float best = 1e30f;
    for (int r = 0; r < rounds; ++r) {
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

void launch_tc(const half* A, const half* B, half* C)
{
    gemm_tc_kernel<<<GRID, BLOCK>>>(A, B, C, M, K, N);
}
void launch_tc5a(const half* A, const half* B, half* C)
{
    gemm_tc5a_kernel<<<GRID, BLOCK>>>(A, B, C, M, K, N);
}

int main()
{
    size_t bytesA = (size_t)M * K * sizeof(half);
    size_t bytesB = (size_t)K * N * sizeof(half);
    size_t bytesC = (size_t)M * N * sizeof(half);

    half *dA, *dB, *dC0, *dC1;
    CHECK(cudaMalloc(&dA, bytesA));
    CHECK(cudaMalloc(&dB, bytesB));
    CHECK(cudaMalloc(&dC0, bytesC));
    CHECK(cudaMalloc(&dC1, bytesC));

    int threads = 256;
    fill_kernel<<<(M * K + threads - 1) / threads, threads>>>(dA, M * K, 0);
    fill_kernel<<<(K * N + threads - 1) / threads, threads>>>(dB, K * N, 1);
    CHECK(cudaDeviceSynchronize());

    launch_tc(dA, dB, dC0);
    launch_tc5a(dA, dB, dC1);
    CHECK(cudaDeviceSynchronize());

    // 校验：5a 与基线逐元素比较
    half *h0 = (half*)malloc(bytesC), *h1 = (half*)malloc(bytesC);
    CHECK(cudaMemcpy(h0, dC0, bytesC, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h1, dC1, bytesC, cudaMemcpyDeviceToHost));
    float max_abs = 0.f;
    for (int i = 0; i < M * N; ++i) {
        float d = fabsf(__half2float(h0[i]) - __half2float(h1[i]));
        if (d > max_abs) max_abs = d;
    }
    printf("max |tc - tc5a| = %.6e   %s\n", max_abs, max_abs < 1e-2f ? "[OK]" : "[WRONG]");

    const double FLOPS = 2.0 * M * N * K;

    // 交叉测量：两个 kernel 轮流测，每个取多轮最小值。
    // 单次顺序测量会被「先跑的冷、后跑的热」影响，实测差过 40%。
    float ms0 = 1e30f, ms1 = 1e30f;
    for (int round = 0; round < 3; ++round) {
        float a = bench(launch_tc,   dA, dB, dC0, 100, 1, 100);
        float b = bench(launch_tc5a, dA, dB, dC1, 100, 1, 100);
        if (a < ms0) ms0 = a;
        if (b < ms1) ms1 = b;
    }

    printf("台阶4  （逐元素装载） %8.4f ms   %6.2f TFLOPS\n", ms0, FLOPS / (ms0 * 1e-3) / 1e12);
    printf("台阶5a（uint4 装载） %8.4f ms   %6.2f TFLOPS   %+.1f%%\n", ms1,
           FLOPS / (ms1 * 1e-3) / 1e12, (ms0 / ms1 - 1.0f) * 100.0f);
    return 0;
}
