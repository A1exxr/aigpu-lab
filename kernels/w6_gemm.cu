#include <torch/extension.h>
#include <cuda_fp16.h>


// 台阶1 naive
// 这个naive虽然也分了tile，但是没有让它们共享数据
__global__ void gemm_naive_kernel(const half* __restrict__ A,
                            const half* __restrict__ B,
                            half* __restrict__ C,
                            int M, int K, int N){
    // 一个 block 负责 C 的一个 32×32 tile，block 内每个线程负责 tile 里的一个元素
    int row = blockIdx.y * blockDim.y + threadIdx.y;        // row 是 M 方向的全局行索引(A 和 C 的 M 方向都可以对应)
    int col = blockIdx.x * blockDim.x + threadIdx.x;        // col 是 N 方向的全局列索引
    if(row < M && col < N)
    {
        // 卡住了写不出来
        // C[row][col] = Σ  A[row][k] × B[k][col]        k = 0 … K-1   就是要表达这个公式
        float acc = 0.f;
        for(int k = 0; k < K; ++k)
        {
            acc += __half2float(A[row * K + k]) * __half2float(B[k * N + col]);
        }
        C[row * N + col] = __float2half(acc);
    }
};


// 台阶2  1个线程算一个C元素，并分块进行
// 现在是tile，A 的 32×32 块、B 的 32×32 块搬进共享内存，线程反复读读数据计算C
// block(bx=2, by=3)  →  负责 C 的第 3 行块、第 2 列块，即 C[96:128, 64:96]
// 需要的数据：A[96:128, :] 的每一段、B[:, 64:96] 的每一段
// 1 个线程算 1 个 C 元素
__global__ void gemm_tiled_kernel(const half* __restrict__ A,
const half* __restrict__ B,
half* __restrict__ C,
int M, int K, int N){
    int row = blockIdx.y * blockDim.y + threadIdx.y;        // 这种索引，也是标准公式，有一个前提，就是一个线程处理1个元素，block正好负责1个C块
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    // 分块大小的都是32x32
    __shared__ half As[32][32];
    __shared__ half Bs[32][32];
    float acc = 0.f;
    
    // 沿着K轴滑动，取A、B块
    // 一个block负责的是一个C块，比如求C块的左上角分块阵，也是32*32的
    // 求这个C块，A在K方向就有1024/32个分块阵，B在K方向也有这么多分块阵
    // 把A和B的这些分块阵，乘加起来，就是C的这个分块阵，所有乘加操作，都是一个block完成的
    // 一个线程只负责 C 的一个元素 C[row, col]  
    for(int k0 = 0; k0 < K; k0 += 32)               // 遍历 K 维度
    {   
        // A 是 [M, K]，tile 是 [M_block, K_block]：
        // A[row_block : row_block+32, k : k+32]
        //   ↑ 行 = M 方向             ↑ 列 = K 方向
        As[threadIdx.y][threadIdx.x] = A[row * K + (k0 + threadIdx.x)];             // 取A、B块

        // B 是 [K, N] 一个 block 要加载 B 的一个 32×32 tile：
        // B[k : k+32, col_block : col_block+32]
        //   ↑ 行 = K 方向        ↑ 列 = N 方向
        Bs[threadIdx.y][threadIdx.x] = B[(k0 + threadIdx.y) * N + col]; 
        __syncthreads();        // 写了共享内存，加屏障

        // 取完块后，沿着K轴计算
        for(int kk = 0; kk < 32; kk++)              // 在当前块进行乘加循环
        {
            // acc += As[ty][kk] * Bs[kk][tx];
            acc += __half2float(As[threadIdx.y][kk]) * __half2float(Bs[kk][threadIdx.x]);
        }
        __syncthreads();
    }
    
    C[row * N + col] = __float2half(acc);
};



// 台阶3 ———— 让每个线程算 16 个 C，把共享内存的读次数降到 1/4。
__global__ void gemm_reg_kernel(const half* __restrict__ A,
const half* __restrict__ B,
half* __restrict__ C,
int M, int N, int K){
    int row = blockIdx.y * 32 + threadIdx.y * 4;            // 全局行起点
    int col = blockIdx.x * 32 + threadIdx.x * 4;            // 全局列起点
    // 两者相加，就是这个线程负责的 4×4 子块左上角的全局坐标

    __shared__ half As[32][32];
    __shared__ half Bs[32][32];
    // 一个元素现在要负责4*4个C元素，所以要维护16个累加器
    // 与 acc[16]的作用一样，区别就是可读性
    // i 对应 A 行偏移量，j 对应 B 列偏移量
    float acc[4][4] = {0.f};                // acc[i][j] 对应 C[row0+i][col0+j]

    for(int k0 = 0; k0 < K; k0 += 32)       // 沿着K方向滑动
    {      
        // 64个线程读取1024个元素  
        // 每个线程都要遍历1024/64 = 16个idx
        // 而且用idx线性遍历，同一warp内idx连续，c也连续，As[]地址也连续，合并访存
        for(int idx = threadIdx.y * 8 + threadIdx.x; idx < 1024; idx += 64)     // 把64个线程的二维坐标压成一维线性编号，然后用64为步长遍历1024个元素
        {
            // 因为idx是按行主序展开的0 1 2...31 / 32 33 ... 63 / 64... 1023
            // idx = r * 32 + c
            int r = idx / 32;       // tile 内的行，0~31     完整跳过了几整行
            int c = idx % 32;       // tile 内的列, 0~31     当前行内第几个

            // 关于A和B的全局索引
            // A 的 M 维度就是行，全局行号 = blockIdx.y * 32 + r
            // A 的 K 维度就是列，全局列号 = k0 + c
            As[r][c] = A[(blockIdx.y * 32 + r) * K + (k0 + c)];

            // B 的 K 维度就是行，全局行号 = k0 + r
            // B 的 N 维度是列，k0的变化不影响全局列号，只有线程索引影响
            // B 的 全局列号 = blockIdx.x * 32 + c
            Bs[r][c] = B[(k0 + r) * N + (blockIdx.x * 32 + c)];

            // 总结：
            // 一个 block 固定负责 C 的一个 32×32 块（M、N 固定），沿 K 方向滑动
            // 每轮装载 A 的 32×32（M 固定、K 变化）和 B 的 32×32（K 变化、N 固定），累加到acc
            // 固定维度（M、N）的全局下标 = blockIdx 决定块起点 + 线程/局部索引决定块内偏移
            // 滑动维度（K）的全局下标 = k0 决定当前分块起点 + 局部索引决定块内偏移
        }
        __syncthreads();   // 写了共享内存，设屏障

        // 内层计算
        for(int kk = 0; kk < 32; ++kk)      // 遍历tile的32个位置，kk对应K下标
        {
            // 每个线程负责一个 4*4 的小块
            float a[4], b[4];
            // 读 4 个 A 元素
            // threadIdx.y * 4 + i	这个线程负责的 4 行（tile 内局部行号）
            for(int i = 0; i < 4; ++i) a[i] = __half2float(As[threadIdx.y * 4 + i][kk]);

            // 读 4 个B 元素
            // threadIdx.x * 4 + j	这个线程负责的 4 列（tile 内局部列号）
            for(int j = 0; j < 4; ++j) b[j] = __half2float(Bs[kk][threadIdx.x * 4 + j]);

            // 16次FMA
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

    // 写回 16 个值
    for(int i = 0; i < 4; ++i)
    {
        for(int j = 0; j < 4; ++j)
        {
            C[(row + i) * N + col + j] = __float2half(acc[i][j]);
        }
    }

}











torch::Tensor gemm(torch::Tensor A, torch::Tensor B){
    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::empty({M, N}, A.options());
    dim3 block(32, 32);                     // 一个 block 有 32×32 = 1024 个线程
    dim3 grid(N/32, M/32);                  // grid.x 沿 N（列），grid.y 沿 M（行）； 总 block 数 = (N/32) × (M/32)
    gemm_naive_kernel<<<grid, block>>>(
        reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(B.data_ptr<at::Half>()),
        reinterpret_cast<half*>(C.data_ptr<at::Half>()),
        M, K, N
    );
    return C;
}


torch::Tensor gemm_tiled(torch::Tensor A, torch::Tensor B){
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    auto C = torch::empty({M, N}, A.options());
    dim3 block(32, 32);
    dim3 grid(N/32, M/32);       
    gemm_tiled_kernel<<<grid, block>>>(
        reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(B.data_ptr<at::Half>()),
        reinterpret_cast<half*>(C.data_ptr<at::Half>()),
        M, K, N
    );

    return C;
}


torch::Tensor gemm_reg(torch::Tensor A, torch::Tensor B){
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    auto C = torch::empty({M, N}, A.options());

    // 一个 block 还是负责 C 的 32×32 块
    // block 现在是 8*8=64个线程，一个线程处理 以各 4*4 的小块，共64个小块
    dim3 block(8, 8);           
    dim3 grid(N/32, M/32);
    gemm_reg_kernel<<<grid, block>>>(
        reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(B.data_ptr<at::Half>()),
        reinterpret_cast<half*>(C.data_ptr<at::Half>()),
        M, K, N
    );

    return C;
}