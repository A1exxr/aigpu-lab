#include <torch/extension.h>
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;         // WMMA 的命名空间


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



// 台阶3 ———— 让每个线程算 16 个 C，把共享内存的读次数降到 1/4 ———— FFMA（普通乘加），一次 1 次乘加
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
        // 线程 0：idx = 0, 64, 128, ..., 960（16 个）
        // 线程 1：idx = 1, 65, 129, ..., 961
        // 线程 63：idx = 63, 127, 191, ..., 1023
        // 而且用idx线性遍历，同一warp内idx连续，c也连续，As[]地址也连续，合并访存
        for(int idx = threadIdx.y * 8 + threadIdx.x; idx < 1024; idx += 64)     // 把64个线程的二维坐标压成一维线性编号，然后用64为步长遍历1024个元素
        {
            // idx只是确定是第几个元素，r 和 c 才是“在 tile 的哪一行哪一列”。
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

            // idx 循环一次，同时写 As[r][c] 和 Bs[r][c] 两个位置。1024 次迭代，As 和 Bs 各自被填满 1024 个位置
        }
        __syncthreads();   // 写了共享内存，设屏障

        // 内层计算
        for(int kk = 0; kk < 32; ++kk)      // 遍历tile的32个位置，kk对应K下标
        {
            // 每个线程负责一个 4*4 的小块
            float a[4], b[4];
            // 读 4 个 A 元素 ———— 4×1 列向量  —— 4 行（ty*4+i）、第 kk 列
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



// 台阶4  ———— 一整个 warp（32 个线程）协作算一块 16×16 —————— 采用WMMA（张量核心），1 个 warp 各管一个 16×16，Tensor Core 一次算完 16×16×16 = 4096 次乘加
// 一个block 有 4 个warp， 处理32*32的C块
__global__ void gemm_tc_kernel(const half* __restrict__ A,
                                const half* __restrict__ B,
                                half* __restrict__ C,
                                int M, int K, int N)
{
    int tx = threadIdx.x;               // 0..31   tile 内的列偏移
    int ty = threadIdx.y;               // 0..3    = warp 编号
    int btid = ty * 32 + tx;            // 0..127  线性线程号（搬数据时用）   block: 32 * 4

    // 在 C 的32*32tile里的布局
    int warp_m = ty / 2;                // 0 1
    int warp_n = ty % 2;                // 0 1
    int bm = blockIdx.y;                // C tile 的行号
    int bn = blockIdx.x;                // C tile 的列号

    __shared__ half As[32][32];
    __shared__ half Bs[32][32];
    
    // 需要累加器是因为，store_matrix_sync只能写到float的共享内存，不能直接写到 half 的 C
    __shared__ float Cf[32][32];        // WMMA 的累加器用float类型


    // WMMA fragment 声明
    // fragment 是WMM的寄存器抽象，一个fragment代表一个16*16的矩阵块，分布在32个lane的寄存器里
    // 16*16*16是标准形状，16*16的输出，K方向16个元素
    // M N K 形状是Tensor Core的物理规格，K方向一次性只能取16个
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16 ,16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);      // 初始化累加器为0

    // 外层K循环
    for(int k0 = 0; k0 < K; k0 += 32)
    {
        // 128线程装1024个元素，1个线程装8个元素
        for(int i = btid; i < 1024; i += 128)
        {
            // 8轮迭代，每轮128个线程，搬128个数据，4 个 warp ———— 每个 warp 负责一整行，r 固定，c 从 0 到 31 连续   
            int r = i / 32;         // 0 1 2 3 ... 31    i是从0到1024
            int c = i % 32;         // 0 1 2 3 ... 31
            As[r][c] = A[(bm * 32 + r) * K + (k0 + c)];
            Bs[r][c] = B[(k0 + r) * N + (bn * 32 + c)];
        }
        __syncthreads();

        // // WMMA 加载与矩阵乘
        // // K 方向 0 ... 15
        // // load_matrix_sync把 16*16 从共享内存上加载到寄存器上
        // // warp_m = 0：从 As[0][0] 开始，取 As[0:16, 0:16]；
        // // warp_m = 1：从 As[16][0] 开始，取 As[16:32, 0:16]
        // // A的16行由warp_m决定，列（K方向）由tile本身决定
        // // ldm = 32，领先维度
        // wmma::load_matrix_sync(a_frag, &As[warp_m * 16][0], 32);    

        // // warp_n = 0：从 Bs[0][0] 开始，取 Bs[0:16, 0:16]；
        // // warp_n = 1：从 Bs[0][16] 开始，取 Bs[0:16, 16:32]。
        // wmma::load_matrix_sync(b_frag, &Bs[0][warp_n * 16], 32);

        // // mma_sync 执行：c_frag = a_frag × b_frag + c_frag  
        // // 也就是 c_frag += a_frag * b_frag;
        // wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        // // K 方向 16..31
        // wmma::load_matrix_sync(a_frag, &As[warp_m * 16][16], 32);
        // wmma::load_matrix_sync(b_frag, &Bs[16][warp_n * 16], 32);
        // wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);


        // 写成循环
        for (int kk = 0; kk < 32; kk += 16) 
        {             
            wmma::load_matrix_sync(a_frag, &As[warp_m * 16][kk], 32);
            wmma::load_matrix_sync(b_frag, &Bs[kk][warp_n * 16], 32);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }

        __syncthreads();   // 下一轮要覆盖 As/Bs
    }

    // 写回累加器共享内存
    wmma::store_matrix_sync(&Cf[warp_m * 16][warp_n * 16], c_frag, 32, wmma::mem_row_major);
    __syncthreads();

    for(int i = btid; i < 1024; i += 128)
    {
        int r = i / 32;
        int c = i % 32;
        C[(bm * 32 + r) * N + bn * 32 + c] = __float2half(Cf[r][c]);
    }
}   


// 台阶4 ———— 优化成 处理 64*64 的 C块
// 现在一个warp还是处理 16 * 16 的C块， 现在需要16个warp才能处理一个 64 * 64的C块，所以一个block有16个warp，512个线程
// grid形状(32,16) = 512 线程 = 16 warp
__global__ void gemm_tc2_kernel(const half* __restrict__ A,
const half* __restrict__ B,
half* __restrict__ C,
int M, int K, int N){
    int tx = threadIdx.x;                   // 0 1 2 ... 31
    int ty = threadIdx.y;                   // 0 1 2 ... 15       也可以看作 warp 编号
    int btid = ty * 32 + tx;                // 0 1 2 ... 511

    // 在 C 的 64*64 tile里的布局，一共有 16 个 16*16 小块
    int warp_m = ty / 4;                    // 0 1 2 3 
    int warp_n = ty % 4;                    // 0 1 2 3

    __shared__ half As[64][32];
    __shared__ half Bs[32][64];
    __shared__ float Cf[64][64];
    

    // WMMA fragment声明
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);


    // 沿K方向推进，K的分块维度是多少， k0 += 多少
    for(int k0 = 0; k0 < K; k0 += 32)
    {
        // 512 个线程装 4096 个元素，1个线程装8个元素(A4个，B4个)
        // 但这里A分块是(64, 32)  B分块是(32, 64)
        // A的行宽是32，B的行宽是64，行宽是一行有多少元素，就是领先维度
        // 而 i = r * 行宽 + c ———— 行宽不一样，不能放到一个循环里
        // r = i / 行宽
        // c = i % 行宽

        // 先加载A的分块阵
        // 这里的2048就是 64*32是分块A的总元素，每个线程搬4个A元素
        for(int i = btid; i < 2048; i += 512)
        {
            int r = i / 32;
            int c = i % 32;
            // 这里我写成 blockIdx.y * 32了，不是32，因为blockIdx表示，这是第几块64*64的tile，所以要乘64
            As[r][c] = A[(blockIdx.y * 64 + r) * K + (k0 + c)];     
        }

        // 加载B的分块
        for(int i = btid; i < 2048; i += 512)
        {
            int r = i / 64;
            int c = i % 64;
            Bs[r][c] = B[(k0 + r) * N + (blockIdx.x * 64 + c)];
        }
        __syncthreads();


        for(int kk = 0; kk < 32; kk += 16)
        {
            wmma::load_matrix_sync(a_frag, &As[warp_m * 16][kk], 32);
            wmma::load_matrix_sync(b_frag, &Bs[kk][warp_n * 16], 64);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        __syncthreads();
    }

    // Cf 领先维度是64
    wmma::store_matrix_sync(&Cf[warp_m * 16][warp_n * 16], c_frag, 64, wmma::mem_row_major);
    __syncthreads();

    for(int i = btid; i < 4096; i += 512)
    {   
        // 这里卡了半天，不知道是 i/？？  不知道填多少
        // 看右边，是要把Cf的值存进去，Cf是(64,64)，行宽是64，就填64
        // 左边 C 是二维全局数组：行 = tile 起始行(blockIdx.y*64) + r，列 = tile 起始列 + c
        int r = i / 64;
        int c = i % 64; 
        C[(blockIdx.y * 64 + r) * N + (blockIdx.x * 64 + c)] = __float2half(Cf[r][c]);
    }
}



// gemm nt变体   C=A×B^T ———— cuBLAS 的 NN/NT/TN/TT 四种布局         
/*
    NT 布局：B 存的是 [N][K]，即 Bs[n][k] 在内存里地址 = n*32 + k
    wmma 要的 matrix_b 是 "K×N 的列主序"：逻辑 (k,n) 必须落在 k + n*ldm
    所以 ldm = 32，指针 = &Bs[n_base][k_base] = &Bs[warp_n*16][kk] 
*/
__global__ void gemm_nt_kernel(const half* __restrict__ A,
                                const half* __restrict__ B,
                                half* __restrict__ C,
                                int M, int K, int N)
{
    int tx = threadIdx.x;               // 0..31   tile 内的列偏移
    int ty = threadIdx.y;               // 0..3    = warp 编号
    int btid = ty * 32 + tx;            // 0..127  线性线程号（搬数据时用）   block: 32 * 4

    // 在 C 的32*32tile里的布局
    int warp_m = ty / 2;                // 0 1
    int warp_n = ty % 2;                // 0 1
    int bm = blockIdx.y;                // C tile 的行号
    int bn = blockIdx.x;                // C tile 的列号

    __shared__ half As[32][32];
    __shared__ half Bs[32][32];
    

    __shared__ float Cf[32][32];      

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;   // NT：Bs 存的是 [n][k]，对 B 而言是列主序
    wmma::fragment<wmma::accumulator, 16 ,16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);     

    // 外层K循环
    for(int k0 = 0; k0 < K; k0 += 32)
    {
        // 128线程装1024个元素，1个线程装8个元素
        for(int i = btid; i < 1024; i += 128)
        {
            // 8轮迭代，每轮128个线程，搬128个数据，4 个 warp ———— 每个 warp 负责一整行，r 固定，c 从 0 到 31 连续   
            int r = i / 32;         // 0 1 2 3 ... 31    i是从0到1024
            int c = i % 32;         // 0 1 2 3 ... 31
            As[r][c] = A[(bm * 32 + r) * K + (k0 + c)];
            Bs[r][c] = B[(bn * 32 + r) * K + (k0 + c)];   // NT：r 走 N，c 走 K（B 存成 [N][K] 行主序）
        }
        __syncthreads();

        for (int kk = 0; kk < 32; kk += 16) 
        {             
            wmma::load_matrix_sync(a_frag, &As[warp_m * 16][kk], 32);
            wmma::load_matrix_sync(b_frag, &Bs[warp_n * 16][kk], 32);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }

        __syncthreads();   // 下一轮要覆盖 As/Bs
    }

    // 写回累加器共享内存
    wmma::store_matrix_sync(&Cf[warp_m * 16][warp_n * 16], c_frag, 32, wmma::mem_row_major);
    __syncthreads();

    for(int i = btid; i < 1024; i += 128)
    {
        int r = i / 32;
        int c = i % 32;
        C[(bm * 32 + r) * N + bn * 32 + c] = __float2half(Cf[r][c]);
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


torch::Tensor gemm_tc(torch::Tensor A, torch::Tensor B){
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    auto C = torch::empty({M, N}, A.options());

    // 这里block设置成128个线程，4个warp， (32,4)是为了让ty对应warp_id —— 0 1 2 3
    // 一个tile是32*32，一个warp计算16*16小块
    // 通过ty来对应tile的4个小块，ty=0 左上、ty=1 右上、ty=2 左下、ty=3 右下
    dim3 block(32, 4);              
    dim3 grid(N/32, M/32);

    gemm_tc_kernel<<<grid, block>>>(
        reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(B.data_ptr<at::Half>()),
        reinterpret_cast<half*>(C.data_ptr<at::Half>()),
        M, K, N
    );

    return C;

}


torch::Tensor gemm_tc2(torch::Tensor A, torch::Tensor B){
    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::empty({M, N}, A.options());
    dim3 block(32, 16);             
    dim3 grid(N/64, M/64);
    gemm_tc2_kernel<<<grid, block>>>(
        reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(B.data_ptr<at::Half>()),
        reinterpret_cast<half*>(C.data_ptr<at::Half>()),
        M, K, N);
    return C;
}


torch::Tensor gemm_nt(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0), K = A.size(1);
    int N = B.size(0);                       // ← B 是 [N, K]，不是 [K, N]
    auto C = torch::empty({M, N}, A.options());

    dim3 block(32, 4);                       // 128 线程 = 4 warp（和 gemm_tc 一样）
    dim3 grid((N + 31) / 32, (M + 31) / 32);

    gemm_nt_kernel<<<grid, block>>>(
        reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(B.data_ptr<at::Half>()),
        reinterpret_cast<half*>(C.data_ptr<at::Half>()),
        M, K, N);
    return C;
}