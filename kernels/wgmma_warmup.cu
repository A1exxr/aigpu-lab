/*
    wgmma_warmup.cu —— D5：wgmma 最小单测   
    本机（4060 / sm_89）不能跑，但只要 sm_90a 能编过、SASS 里有 HGMMA，就说明指令生成是对的；  

    判据（注意：SASS 里 wgmma 的助记符是 HGMMA，不是 WGMMA）：
      nvcc -arch=sm_90a -cubin wgmma_warmup.cu
      cuobjdump -sass wgmma_warmup.cubin | grep -cE 'HGMMA|WARPGROUP'
      期望：HGMMA 1、WARPGROUP.ARRIVE 1、WARPGROUP.DEPBAR 1 


    在 Ampere 架构（如 A100）中，Tensor Core 的计算单位是 Warp（32 个线程），且需要手动把数据从 Shared Memory 加载到寄存器（使用 ldmatrix 指令）
    Hopper 架构引入了 WGMMA（Warpgroup Matrix Multiply Accumulate）
        计算单位扩大：以 Warpgroup（4 个 Warp，即 128 个线程） 为基本单位协同执行一条指令
        异步且直接读取 SMEM：它是一条异步指令；更关键的是，矩阵 A 和 B 可以直接从 Shared Memory 读取，完全省去了手动加载到寄存器的步骤（矩阵 B 必须在 SMEM，矩阵 A 可以在 SMEM 或寄存器）
        A矩阵：可以在 register 或 shared memory
        B矩阵：必须在 shared memory
        C结果：一定在 register


    矩阵描述符 Matrix Descriptor，也是 64 位的，告诉 Tensor Core：
        起始地址：数据在 SMEM 的哪个位置
        LBO (Leading Byte Offset)：沿着 K 维度（内积维度），相邻两个“核心矩阵（Core Matrix）”之间的字节距离
        SBO (Stride Byte Offset)：沿着 M 或 N 维度，相邻两个“8 行组”之间的字节距离
            描述符里 LBO/SBO 都要 >> 4（16 字节为单位）
        Swizzle 模式：描述符需要告诉硬件如何正确解交织
        
    异步同步机制：WGMMA 是异步的，发出指令后线程不会立刻停下来等结果，所以必须有一套严格的同步机制
        wgmma.fence：内存栅栏；确保在发起计算前，Shared Memory 里的数据已经完完全全写好了（比如 TMA 刚搬完）
        wgmma.mma_async：真正发起矩阵乘计算
        wgmma.commit_group：把前面发起的异步计算打包成一个“组（Group）”提交给硬件队列
        wgmma.wait_group 0：让线程挂起，等待队列里所有的计算组都执行完毕，此时寄存器里的累加器（Accumulator）结果才是有效的


    形状：C[64][64] = A[64][16] @ B[16][64]，fp16 输入 / fp32 累加
    一个 block = 128 线程 = 1 个 warpgroup（wgmma 的执行单位）
    累加器：64*64/128 = 32 个 float/线程    

    在 wgmma 里面，Tensor Core 不按行/列读数据，而是按照 Core Matrix 读
    对于 fp16，一个 Core Matrix = 8 行 × 8 列 = 8×16B = 128B
    现在 A[M][K]是[64][16]  B[K][N]是[16][64]  均为 row-major
    对于矩阵A：
        一行 16 个 half = 32 字节
        K = 16 ————> 一行有 16/8 = 2 个 Core Matrix
        a_lbo = 一个 Core Matrix 的宽度 = 8 * sizeof(half) = 16字节
        a_sbo = A 矩阵就是 M 方向，跨 8 行的字节距离 = 8 * 16 * 2B = 256字节
    
    对于矩阵B：
        B 矩阵的 Core Matrix 在 K 方向被切成 2 块（k=0–7、k=8–15）
                在 N 方向被切成 8 块（n=0–7、8–15、…、56–63）   一共2*8=16个 Core Matrix
        一行 64 个 half = 128 字节
        N = 64 ————> 一行有 64/8 = 8 个 Core Matrix
        b_lbo = 8 * 64 * 2B  = 1024字节，K 方向下一块8*8的Core Matrix矩阵，跳了8行
        b_sbo = B 矩阵就是 N 方向，下一个 Core Matrix = 8 * 2 = 16字节 


    现在有个问题：A/B 在 smem 里的布局和描述符声明的格式要一致
        但 smem 装载是 row-major，就是逐元素按行主序拷，不满足 wgmma 的布局要求
        wgmma 要求每个 8×8 core matrix 在 smem 里是连续 128 B；row-major 的 [64][16] 里，一个 core matrix 的 8 行分散在 32 B × 8 的区域，中间夹着 k=8..15 的数据
        所以 LBO/SBO 的推导建立在错误前提上，现在算的是 块基址之间的距离

        硬件拿到块基址后，连续读 128 B，当成 8 行 × 8 列 用
            它以为的 128 B              smem 里的前 128 B
            m0: k0..k7 (16B)         m0: k0..k7 (16B) | k8..k15 (16B)
            m1: k0..k7 (16B)         m1: k0..k7 (16B) | k8..k15 (16B)
            ...                      m2: ...
            m7: k0..k7 (16B)         m3: ...

        它读到 4 整行 × 16 列 ，却当 8 行 × 8 列 用：数都在，位置全错，不报错

    解决方法：
        smem扁平声明 As[M * K] Bs[K * N]
        按块装载：把二维坐标(m,k) 换成 块号+块内偏移  ——> 块号 × 64 + 块内偏移
        再改 lbo 和 sbo


    smem 描述符（64 位）：起始地址(>>4) + LBO(>>4) + SBO(>>4) + swizzle 模式
    布局：A 是 M 行 K 列、K 连续；B 是 K 行 N 列、N 连续（SS 变体都从 smem 读）
    顺序：wgmma.fence → mma_async → commit_group → wait_group → 才能读结果
*/



#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define M 64
#define K 16
#define N 64
#define NWARP 4                       // 1 warpgroup = 4 warp
#define NTHREADS (NWARP * 32)

// ---------------------------------------------------------------------------
// 描述符位域（PTX ISA 9.7.16.4.1 matrix descriptor）
//   位  0-13 : start address >> 4        （smem 起始地址 / 16，16 B 为单位）
//   位 16-29 : leading byte offset >> 4  （LBO  Leading Byte Offset ÷ 16）
//   位 32-45 : stride byte offset >> 4   （SBO  Stride Byte Offset ÷ 16）
//   位 49-51 : base offset（SWIZZLE_NONE 时填 0）
//   位 62-63 : swizzle 模式（0 = none, 1 = 128B, 2 = 64B, 3 = 32B）
// ---------------------------------------------------------------------------
// 矩阵描述符 
__device__ __forceinline__ unsigned long long make_desc(const void* p,
                                                        unsigned lbo,
                                                        unsigned sbo,
                                                        unsigned swizzle = 0) 
{
    // 起始地址 p，Tensor Core 去 smem 这个字节位置开始取数据
    unsigned long long addr = (unsigned long long)__cvta_generic_to_shared(p);
    unsigned long long d = 0;
    d |= (addr >> 4) & 0x3FFFu;                                               // 起始地址/16，SMEM地址必须16字节对齐
    d |= ((unsigned long long)(lbo >> 4) & 0x3FFFu) << 16;                    // LBO
    d |= ((unsigned long long)(sbo >> 4) & 0x3FFFu) << 32;                    // SBO
    d |= ((unsigned long long)(swizzle & 0x3u)) << 62;
    return d;
}

// 确保此指令之前所有smem写入对后续wgmma可见
__device__ __forceinline__ void wgmma_fence() {
    // .aligned：warpgroup内 128 线程都到达才继续
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

// 把前面发出的 mma_async 打包成一个“异步组”，提交到硬件执行队列
__device__ __forceinline__ void wgmma_commit() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

// 阻塞当前线程，直到队列中所有已提交的异步组执行完毕（pending groups = 0）
__device__ __forceinline__ void wgmma_wait0() {
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
}

/*
    Hopper 上一条 wgmma 指令的内联汇编封装：一个 warpgroup（128 线程）合作算 C[64][64] = A[64][16] · B[16][64] + C，
    每条 lane 拿 32 个 fp32 累加器  64*64/128 = 32 个元素
    float (&d)[32]  引用传递一个长度 32 的 float 数组——32 个累加器寄存器，不能按值传（数组不能拷贝）
    A 和 B 的两个 64 位 smem 描述符
    结果直接写在 d[0..31] 里，+f 约束是“读写”，指令累加到 d 上

    wgmma：wgmma级矩阵乘加
    .mma_async：异步矩阵乘加：发射后立即返回，不等结果
    .sync：warpgroup 内 128 线程同步进行
    .aligned：所有线程必须执行同一条指令，不能在分支里
    m64n64k16：M=64, N=64, K=16
    f32 D/C的类型
    f16 A类型
    f16 B类型
*/

__device__ __forceinline__ void wgmma_m64n64k16(float (&d)[32],
                                                unsigned long long a_desc,
                                                unsigned long long b_desc) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, "
        "%32, %33, 1, 1, 1, 0, 0;\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]),
          "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]),
          "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]),
          "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
          "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]),
          "+f"(d[30]), "+f"(d[31])
        : "l"(a_desc), "l"(b_desc));
}

// ---------------------------------------------------------------------------
// 单测 kernel
//   smem A: [64][16] half，行跨度 32 B；smem B: [16][64] half，行跨度 128 B
//   描述符里只有 LBO / SBO 需要算：
//     LBO = 沿 K 方向，相邻两个 core matrix 之间的字节距离
//     SBO = 沿 M（A）/ N（B）方向，相邻两个"8 行组"之间的字节距离
//   core matrix（f16）= 8 行 × 8 列 = 8 行 × 16 B
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(NTHREADS) wgmma_warmup_kernel(const half* __restrict__ A, 
                                                                const half* __restrict__ B,
                                                                float* __restrict__ C) 
{
    // 扁平，方便算块地址
    __shared__ alignas(128) half As[M * K];      // 64 * 16 * 2 = 2048 B
    __shared__ alignas(128) half Bs[K * N];      // 16 * 64 * 2 = 2048 B

    const int tid = threadIdx.x;
    for(int i = tid; i < M * K; i += NTHREADS)
    {
        const int m = i / K;
        const int k = i % K;

        // >> 和 & 用于索引拆解、   / 用于维度计算是标准实践
        // 块号：(m >> 3) * (K / 8) + (k >> 3)  m 方向定位第几行块，k 方向定位第几列块
        // 块内偏移：(m & 7) * 8 + (k & 7)   块内 8 行 × 8 列，按行优先
        // 共享内存写入后，重排为 K-major WGMMA 布局；就是对于块号，是先遍历K方向，再遍历M方向，即 K-major 块排列顺序
        As[((m >> 3) * (K / 8) + (k >> 3)) * 64 + (m & 7) * 8 + (k & 7)] = A[i];
        /*
            比如：(17,21)
                Tile 级定位：17 >> 3 = 2 → M 方向第 2 个 8×8 子块，21 >> 3 = 2 → K 方向第 2 个 8×8 子块
                子块内偏移：17 & 7 = 1 → 子块内第 1 行（子块内行号 0~7），21 & 7 = 5 → 子块内第 5 列（子块内列号 0~7）
                最终 SMEM 地址 = [2 * (K/8) + 2] * 64 + 13
        */
    }

    for(int i = tid; i < K * N; i += NTHREADS)
    {
        const int k = i / N;
        const int n = i % N;

       // Bs 块号排列和 As 完全一样，只有块内偏移交换了行/列
       // B 是 N 维连续，所以块内偏移变了：
        Bs[((n >> 3) * (K / 8) + (k >> 3)) * 64 + (k & 7) * 8 + (n & 7)] = B[i];
    }
    __syncthreads();            // 128 个线程全部完成了对 SMEM 的写入操作


   
    const unsigned a_lbo = 128;         // k 加 8（下一列块）→ 块号 +1 → +64 half = 128 B
    const unsigned a_sbo = 256;         // m 加 8（下一行块）→ 块号 +K/8=2 → +128 half = 256 B
    const unsigned b_lbo = 128;        
    const unsigned b_sbo = 256;        

    // 为矩阵 A 和矩阵 B 分别生成一个 64 位的 “硬件描述符”
    unsigned long long a_desc = make_desc(As, a_lbo, a_sbo, 0);
    unsigned long long b_desc = make_desc(Bs, b_lbo, b_sbo, 0);

    float d[32];
    for (int i = 0; i < 32; ++i) d[i] = 0.f;

    // 已经有了__syncthreads()，还是要写wgmma_fence，因为 __syncthreads() 只保证了线程执行顺序，但编译器或硬件可能会把后续的 wgmma.mma_async 指令提前发射
    // wgmma_fence() 作为一个内存栅栏，强制保证在它之前的所有 SMEM 写入操作，对后续的异步 Tensor Core 操作是完全可见的
    wgmma_fence();                        // 确保SMEM数据已写完
    wgmma_m64n64k16(d, a_desc, b_desc);   // 提交计算任务（异步，不等结果）
    wgmma_commit();                       // 打包成组提交到硬件队列
    wgmma_wait0();                        // 阻塞等待所有组完成
    __syncthreads();

    // ---- 累加器 → C 的映射 ------------------------------------
    const int lane    = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;

    for (int j = 0; j < N / 8; ++j) 
    {   
        // 每个 warp 负责 16 行，4 个 warp 覆盖 64 行
        // 每行由 4 个线程负责
        // 每个线程在一个 8×8 Core Matrix 内负责 2 列   这是硬件自己的实现方法
        const int row = warp_id * 16 + lane / 4;
        const int col = j * 8 + (lane % 4) * 2;
        // 每个线程在每个 8×8 Core Matrix 中恰好持有 4 个 float 累加器寄存器
        C[row * N + col]         = d[4*j + 0];
        C[row * N + col + 1]     = d[4*j + 1];
        C[(row + 8) * N + col]     = d[4*j + 2];
        C[(row + 8) * N + col + 1] = d[4*j + 3];
    }
}
