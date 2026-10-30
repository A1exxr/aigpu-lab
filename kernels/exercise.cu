#include<cuda_fp16.h>


__global__ void gemm_tile_kernel(const half* __restrict__ A,
const half* __restrict__ B,
half* __restrict__ C,
int M, int K, int N){
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
    // 一个block 负责 一个 32*32 的C块
    int row = blockIdx.y * 32 + threadIdx.y * 4;    // 当前线程在C块的row坐标
    int col = blockIdx.x * 32 + threadIdx.x * 4;    // 当前线程在C块的col坐标

    __shared__ half As[32][32];
    __shared__ half Bs[32][32];

    float acc[4][4] = {0.f};

    // 一个block 处理1024个元素，这个tile是32*32的，tile里还有小tile
    for(int k = 0; k < K; k += 32)
    {
        for(int idx = threadIdx.y * 8 + threadIdx.x; idx < 1024; idx+=64){
            // 装载A B 块
            // idx在tile里的偏移
            int r = idx / 32;           // idx在32*32的C块里的哪一行
            int c = idx % 32;           // idx在哪一列

            As[r][c] = A[(blockIdx.y * 32 + r) * K + k + c];
            Bs[r][c] = B[(k + r) * N + blockIdx.x * 32 + c];
        }
        __syncthreads();

        for(int kk = 0; kk < 32; ++kk)
        {
             // 从共享内存到寄存器
            float a[4] = {0.f};
            float b[4] = {0.f};
            
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
        __syncthreads();  // 在k循环内部
    }

    
    for(int i = 0; i < 4; ++i)
    {
        for(int j = 0; j < 4; ++j)
        {
            C[(row + i) * N + col + j] = __float2half(acc[i][j]);
        }
    }

}

