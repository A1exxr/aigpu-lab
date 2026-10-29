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
int M, int K, int N){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // 还是一个block处理一个C块，32*32  
    // 现在是一个线程要处理4*4 16个元素  寄存器要放4个值，维护1个寄存器就行了，然后
    float acc[4] = {0.f};
    __shared__ half As[32][32];
    __shared__ half Bs[32][32];


    for(int idx = ty*8+tx; idx < 1024; idx+=64){
        As[ty][tx] = A[row * K + idx];
        
    }
}

