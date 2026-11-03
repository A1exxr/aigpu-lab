#include<cuda_fp16.h>


__global__ void gemm_tile_kernel(const half* __restrict__ A,
                                const half* __restrict__ B,
                                half* __restrict__ C,
                                int M, int K, int N)
{
    // 一个线程算一个C元素，分块32*32
    int col = blockIdx.x * blockDim.x + threadIdx.x;   // 全局列  C的列   B的行
    int row = blockIdx.y * blockDim.y + threadIdx.y;    // 全局行  C的行 A的列

    __shared__ half As[32][32];
    __shared__ half Bs[32][32];
    // 不需要Cs  直接线程的寄存器存着

    float acc = 0.f;
    for(int k = 0; k < K; k += 32){
        As[threadIdx.y][threadIdx.x] = A[row * K + k + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(k + threadIdx.y) * N + col];

        __syncthreads();

        // 加载完就要乘加一次
        #pragma unroll
        for(int kk = 0; kk < 32; ++kk){
            // 这里累加是因为，一个线程管C的一个元素，需要 A 的一行和 B 的一列的 点积
            acc += __half2float(As[threadIdx.y][kk]) * __half2float(Bs[kk][threadIdx.x]);
        }
        __syncthreads();
    }

    if(row < M && col < N){
        C[row * N + col] = __float2half(acc);
    }
}



// gemm 台阶3 寄存器分块
// M = N = K = 1024，fp16
// block(8, 8) = 64 线程；grid(N/32, M/32)
// 每个线程算 4×4 = 16 个 C 元素
__global__ void gemm_reg_kernel(const half* __restrict__ A,
                                const half* __restrict__ B,
                                half* __restrict__ C,
                                int M, int K, int N)
{
    // 这里是32*32小块，每个线程都是其中一个4*4小块
    int row = blockIdx.y * 32 + threadIdx.y * 4;
    int col = blockIdx.x * 32 + threadIdx.x * 4;

    __shared__ half As[32][32];
    __shared__ half Bs[32][32];
    float acc[4][4] = {0.f};

    float a[4] = {0.f};
    float b[4] = {0.f};

    // 沿着K维归约
    for(int k0 = 0; k0 < K; k0 += 32)
    {
        // 把block中的8*8线程id展开到1维；1个block处理1024个元素
        for(int tid = threadIdx.y * 8 + threadIdx.x; tid < 1024; tid += 64)
        {
            int r = tid / 32;      // 线程在32*32tile里的行索引
            int c = tid % 32;

            As[r][c] = A[(blockIdx.y * 32 + r) * K + k0 + c];
            Bs[r][c] = B[(k0 + r) * N + blockIdx.x * 32 + c];            
        }
        __syncthreads();
        // 不能把计算写进装载的循环里
        // 但是装载和计算都是在同一个K块里，先装载，然后就计算
        for(int kk = 0; kk < 32; ++kk)
        {
            // 把值存到寄存器里
            for(int i = 0; i < 4; ++i)
            {
                a[i] = __half2float(As[threadIdx.y * 4 + i][kk]);
            }
            for(int i = 0; i < 4; ++i)
            {
                b[i] = __half2float(Bs[kk][threadIdx.x * 4 + i]);
            }

            for(int i = 0; i < 4; ++i)
            {
                for(int j = 0; j < 4; ++j)
                {
                    acc[i][j] += a[i] * b[j];
                }
            }
        }       
        __syncthreads();
    }

    for(int i = 0; i < 4; ++i)
    {
        for(int j = 0; j < 4; ++j)
        {
            C[(row + i) * N + col + j] = __float2half(acc[i][j]);
        }
    }

}

#include<mma.h>
using namespace nvcuda;

__global__ void gemm_tc_kernel(const half* __restrict__ A,
                                const half* __restrict__ B,
                                half* __restrict__ C,
                                int M, int K, int N)
{
    // grid(32,32)
    // 一个block负责32*32的C块，4个warp，1个warp负责一个16*16的小块，通过tensor core
    // block是4*32=128线程
    int tx = threadIdx.x;       // 0 1 2 ... 31
    int ty = threadIdx.y;       // 0 1 2 3 
    int bid = ty * 32 + tx;     // 0 1 2 ... 127

    int warp_m = ty / 2;        // 在32*32C块里 warp处理的16*16小块的行索引
    int warp_n = ty % 2;    

    int cm = blockIdx.y;        // C块的行索引
    int cn = blockIdx.x;

    __shared__ half As[32][32];
    __shared__ half Bs[32][32];
    __shared__ float Cf[32][32];

    wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> a_reg;
    wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> b_reg;
    wmma::fragment<wmma::accumulator,16,16,16,float> c_reg;
    wmma::fill_fragment(c_reg, 0.0f);


    // 沿着K维归约
    for(int k0 = 0; k0 < K; k0 += 32)
    {
        // 在32*32的C块里，有4个16*16的小块
        for(int i = bid; i < 1024; i += 128)
        {
            int r = bid / 32;
            int c = bid % 32;

            As[r][c] = A[(cm * 32 + r) * K + k0 + c];
            Bs[r][c] = B[(k0 + r) * N + cn * 32 + c];
        }
        __syncthreads();

        for(int kk = 0; kk < 32; kk += 16)
        {
            wmma::load_matrix_sync(a_reg, &As[warp_m * 16][kk], 32);
            wmma::load_matrix_sync(b_reg, &Bs[kk][warp_n * 16], 32);
            wmma::mma_sync(c_reg, a_reg, b_reg, c_reg);
        }
        __syncthreads();
    }

    // 写回累加器
    wmma::store_matrix_sync(&Cf[warp_m * 16][warp_n * 16], c_reg, 32, wmma::mem_row_major);
    __syncthreads();

    for(int i = bid; i < 1024; i += 128)
    {
        int r = bid / 32;
        int c = bid % 32;
        C[(blockIdx.y * 32 + r) * N + blockIdx.x * 32 + c] = __float2half(Cf[r][c]);
    }
}


// 写一个 tensor core 矩阵乘
__global__ void gemm_tc_kernel(const half* __restrict__ A,
const half* __restrict__ B,
half* __restrict__ C,
int M, int K, int N){
    // C块 32 * 32    一个warp处理16*16的小块，4个warp处理一个C块，4个warp是一个block
    // block是(32,4)  规模
    int tx = threadIdx.x;       // 0 1 2 ... 31
    int ty = threadIdx.y;       // 0 1 2 3
    int bid = ty * 32 + tx;     // 0 1 2 ... 127        128个线程

    int warp_m = ty / 2;        // warp在C块里的行索引 4个warp
    int warp_n = ty % 2;

    __shared__ half As[32][32];
    __shared__ half Bs[32][32];
    __shared__ float Cf[32][32];        // 累加器

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_freg;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);


    // 沿着K维归约
    for(int k0 = 0; k0 < K; k0 += 32)
    {
        // 装载数据到共享内存
        for(int i = bid; i < 1024; i += 128)
        {
            int r = i / 32;
            int c = i % 32;
            As[r][c] = A[(blockIdx.y * 32 + r) * K + k0 + c];
            Bs[r][c] = B[(k0 + r) * N + blockIdx.x * 32 + c];
        }
        __syncthreads();

        for(int kk = 0; kk < 32; kk += 16)
        {
            wmma::load_matrix_sync(a_frag, &As[warp_m * 16][kk], 32);
            wmma::load_matrix_sync(b_freg, &Bs[kk][warp_n * 16], 32);
            wmma::mma_sync(c_frag, a_frag, b_freg, c_frag);
        }
        __syncthreads();

    }
    wmma::store_matrix_sync(&Cf[warp_m*16][warp_n*16], c_frag, 32, wmma::mem_row_major);
    __syncthreads();

    for(int i = bid; i < 1024; i += 1024)
    {
        int r = i / 32;
        int c = i % 32;
        C[(blockIdx.y * 32 + r) * N + blockIdx.x * 32 + c] = __float2half(Cf[r][c]);
    }
}