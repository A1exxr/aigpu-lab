/*
    1、线程与片段划分
        block 64×64，4 个 warp（128 线程）排成 2×2，每个 warp 算 32×32
        每 warp：m 方向 2 个 16、n 方向 4 个 8 → acc[2][4][4] = 32 个 float 累加器
        片段映射自己摆（不用 wmma）

    2、mma.sync 一条顶 8192 条 FFMA

    3、XOR swizzle

    4、cp.async 双缓冲
*/
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define BM 64           // block 覆盖的 M（试「小 tile + 高占用」：尾波效率 67% → 89%）
#define BN 64           // block 覆盖的 N
#define BK 32           // 一个 stage 的 K
#define STAGES 2        // cp.async 缓冲级数
#define WARP_M 32       // 每个 warp 算的 M
#define WARP_N 32       // 每个 warp 算的 N
#define MT (WARP_M / 16)                  // 4
#define NT (WARP_N / 8)                   // 2
#define WM_DIM (BM / WARP_M)              // M 方向几个 warp = 2
#define WN_DIM (BN / WARP_N)              // N 方向几个 warp = 4
#define SM_THREADS (WM_DIM * WN_DIM * 32) // 8 个 warp = 256 线程

#define APAD BK         // A 每行 BK 个 half（不 padding，靠 swizzle）
#define BPAD BN         // B 每行 BN 个 half
#define ASZ (BM * APAD) // 一个 stage 的 A 元素数（128×32）
#define BSZ (BK * BPAD) // 一个 stage 的 B 元素数（32×64）

// XOR swizzle：以 8 half（16 B）为一组，组号与行号异或
__device__ __forceinline__ int swz_a(int m, int g) { return m * APAD + ((g ^ (m & 3)) << 3); }
__device__ __forceinline__ int swz_b(int k, int g) { return k * BPAD + ((g ^ (k & 7)) << 3); }

__device__ __forceinline__ unsigned smem_addr(const void* p) {
    return (unsigned)__cvta_generic_to_shared(p);
}

__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_addr(dst)), "l"(src));
}

__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n"); }

template <int N_>
__device__ __forceinline__ void cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N_));
}

// A 片段：m16k16 → 4 个寄存器（ldmatrix.x4，不转置）
__device__ __forceinline__ void ldm_a(const half* base, unsigned (&r)[4]) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(smem_addr(base)));
}

// B 片段：k16n8 → 2 个寄存器（ldmatrix.x2.trans，从 [k][n] 布局转成 col 片段）
__device__ __forceinline__ void ldm_b(const half* base, unsigned (&r)[2]) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(r[0]), "=r"(r[1])
        : "r"(smem_addr(base)));
}

// D = A·B + D（m16n8k16，fp16 输入 fp32 累加）
__device__ __forceinline__ void mma16816(float (&d)[4], const unsigned (&a)[4], const unsigned (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void load_stage(half* As, half* Bs,
                                           const half* __restrict__ A, const half* __restrict__ B,
                                           int N, int K, int by, int bx, int k0, int tid) {
    // A：BM 行 × BK 列，每行 BK/8 = 4 组 → 512 组
    for (int idx = tid; idx < BM * (BK / 8); idx += SM_THREADS) {
        const int m = idx / (BK / 8);
        const int g = idx % (BK / 8);
        const int gm = by * BM + m;
        cp_async16(As + swz_a(m, g), A + (size_t)gm * K + k0 + g * 8);
    }
    // B：BK 行 × BN 列，每行 BN/8 = 8 组 → 256 组
    for (int idx = tid; idx < BK * (BN / 8); idx += SM_THREADS) {
        const int k = idx / (BN / 8);
        const int g = idx % (BN / 8);
        cp_async16(Bs + swz_b(k, g), B + (size_t)(k0 + k) * N + bx * BN + g * 8);
    }
}

__global__ void __launch_bounds__(SM_THREADS)
gemm_mma_kernel(const half* __restrict__ A, const half* __restrict__ B, half* __restrict__ C,
                int M, int N, int K) {
    extern __shared__ half smem[];
    half* As = smem;                // [STAGES][BM][APAD]
    half* Bs = smem + STAGES * ASZ; // [STAGES][BK][BPAD]

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int by = blockIdx.y;
    const int bx = blockIdx.x;

    const int wm = (warp_id / WN_DIM) * WARP_M;
    const int wn = (warp_id % WN_DIM) * WARP_N;

    float acc[MT][NT][4];
    #pragma unroll
    for (int i = 0; i < MT; ++i)
        #pragma unroll
        for (int j = 0; j < NT; ++j)
            #pragma unroll
            for (int c = 0; c < 4; ++c) acc[i][j][c] = 0.f;

    const int nk = K / BK;
    int buf = 0;

    load_stage(As, Bs, A, B, N, K, by, bx, 0, tid);
    cp_commit();

    for (int kb = 0; kb < nk; ++kb) {
        const int next = kb + 1;
        if (next < nk) {
            load_stage(As + (buf ^ 1) * ASZ, Bs + (buf ^ 1) * BSZ, A, B, N, K, by, bx, next * BK, tid);
            cp_commit();
            cp_wait<1>();   // 放掉一个 → 当前 stage 就绪
        } else {
            cp_wait<0>();
        }
        __syncthreads();

        const half* Asb = As + buf * ASZ;
        const half* Bsb = Bs + buf * BSZ;

        #pragma unroll
        for (int kk = 0; kk < BK / 16; ++kk) {   // 一个 stage 里 2 个 k 段
            const int k0 = kk * 16;

            unsigned a[MT][4];
            #pragma unroll
            for (int i = 0; i < MT; ++i) {
                const int mrow = wm + i * 16 + (lane & 15);
                const int kg = (k0 >> 3) + (lane >> 4);
                ldm_a(Asb + swz_a(mrow, kg), a[i]);
            }

            unsigned b[NT][2];
            #pragma unroll
            for (int j = 0; j < NT; ++j) {
                const int krow = k0 + (lane & 15);
                const int ng = (wn >> 3) + j;
                ldm_b(Bsb + swz_b(krow, ng), b[j]);
            }

            #pragma unroll
            for (int i = 0; i < MT; ++i)
                #pragma unroll
                for (int j = 0; j < NT; ++j) mma16816(acc[i][j], a[i], b[j]);
        }

        __syncthreads();   // 算完才能覆盖这个 stage
        buf ^= 1;
    }

    // epilogue：每行 2 个 half 合成 32 位存
    const int col0 = bx * BN + wn + (lane & 3) * 2;
    const int row0 = by * BM + wm + (lane >> 2);
    #pragma unroll
    for (int i = 0; i < MT; ++i) {
        #pragma unroll
        for (int j = 0; j < NT; ++j) {
            const int r = row0 + i * 16;
            const int c = col0 + j * 8;
            *reinterpret_cast<__half2*>(C + (size_t)r * N + c) =
                __floats2half2_rn(acc[i][j][0], acc[i][j][1]);
            *reinterpret_cast<__half2*>(C + (size_t)(r + 8) * N + c) =
                __floats2half2_rn(acc[i][j][2], acc[i][j][3]);
        }
    }
}

// ---------------- host ----------------

#include <torch/extension.h>

torch::Tensor gemm(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "A/B 必须在 GPU 上");
    TORCH_CHECK(A.dtype() == torch::kHalf && B.dtype() == torch::kHalf, "只支持 fp16");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "只支持 2 维");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "A/B 必须连续");

    const int M = A.size(0), K = A.size(1);
    const int N = B.size(1);
    TORCH_CHECK(B.size(0) == K, "K 不匹配");
    TORCH_CHECK(M % BM == 0 && N % BN == 0 && K % BK == 0,
                "本版要求 M、N 能被 tile 整除，K%32==0");

    auto C = torch::empty({M, N}, A.options());

    const int smem_bytes = STAGES * (ASZ + BSZ) * sizeof(half);
    dim3 grid(N / BN, M / BM);
    gemm_mma_kernel<<<grid, SM_THREADS, smem_bytes>>>(
        reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(B.data_ptr<at::Half>()),
        reinterpret_cast<half*>(C.data_ptr<at::Half>()), M, N, K);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "kernel 启动失败: ", cudaGetErrorString(err));
    return C;
}
