/*
    mma_warmup.cu —— D2 的第一个单测：用一个 warp 算 16x16 · 16x8 → 16x8

    目的：把 m16n8k16 的寄存器布局和 ldmatrix 装料"落到手上"，
         

    ---- 一张图（fp16 进、fp32 累加）----
      A [16][16]  ──ldmatrix.x4──►  a0..a3（4 个 32 位寄存器 = 8 个 half）
      B [16][8]   ──转置进 smem──►  BsT[8][16] ──ldmatrix.x2──►  b0,b1（4 个 half）
      mma.sync m16n8k16 → c0..c3（4 个 float）

    ---- A 的 8 个 half ----
      g = lane / 4              行号 0 ... 7
      t = (lane % 4) * 2        起始列 0, 2, 4, 6
      逻辑上，每条 lane 拿 A 的 8 个 half，记作 h0..h7
      h0 = A[g  ][t]       h1 = A[g  ][t+1]
      h2 = A[g+8][t]       h3 = A[g+8][t+1]
      h4 = A[g  ][t+8]     h5 = A[g  ][t+9]
      h6 = A[g+8][t+8]     h7 = A[g+8][t+9]

      具体例子：
        对于 lane 0：g = 0、t = 0    
        8 个 half 的位置：
          A[0][0], A[0][1], A[8][0], A[8][1], A[0][8], A[0][9], A[8][8], A[8][9]
        4 个寄存器里装的：
          a0={A[0][0],A[0][1]}, a1={A[8][0],A[8][1]}, a2={A[0][8],A[0][9]}, a3={A[8][8],A[8][9]}

        对于 lane 1：g = 0、t = 2
        8 个 half 的位置：
          A[0][2], A[0][3], A[8][2], A[8][3], A[0][10], A[0][11], A[8][10], A[8][11]
        4 个寄存器里装的：
          a0={A[0][2],A[0][3]}, …

      4 条 lane（0..3）覆盖第 0 行 
      4 条 lane（4..7）覆盖第 1 行
      ...
      4 条 lane（28..31）覆盖第 7 行
      32 lane × 8 half = 256 half = 16×16，正好铺满

      为什么是 +8行  +8列
      A[16][16]的布局如下：
      
      ┌─────────────┬─────────────┐
      │ 块0         │ 块2         │
      │ A[0..7]     │ A[0..7]     │
      │ [0..7]      │ [8..15]     │
      ├─────────────┼─────────────┤
      │ 块1         │ 块3         │
      │ A[8..15]    │ A[8..15]    │
      │ [0..7]      │ [8..15]     │
      └─────────────┴─────────────┘
      每条 lane 从 4 个块各拿 2 个 half：  a0 装块0 的 2 half，a1 装块1 的 2 half，a2 装块2 的 2 half，a3 装块3 的 2 half。 4 个寄存器对应 4 个块
      16×16 被切成 4 个 8×8 块，不是 16 个 4×4 块。 +8 是“跳到下一个 8×8 块”的偏移
      A 和 C 的布局规律完全一样

    ---- B 的 4 个 half（B 是 k×n，mma 要 col 布局）----
      g = lane / 4           列号（N 维，0 ... 7）
      t = (lane % 4) * 2     行号（K 维，0, 2, 4, 6）

      B 形状是 [16][8] 所以拆成两个 8*8 的块， +8只跳K维

      b0,b1 = B[t  ][g], B[t+1][g]          ← 同一列、相邻两行
      b2,b3 = B[t+8][g], B[t+9][g]

      原始 B 是[K][N]行主序：B[t][g] 和 B[t+1][g] 在内存里差 8（因为每行 8 列），不连续
      而 ldmatrix 一次只能搬连续 8 个 half，所以必须把 B 转置成 BsT[n][k] = B[k][n]
      smem 物理上没有行列语义，ldmatrix 要求每条 lane 给一个地址 pb，硬件从 pb 开始，物理上连续 8 个 half，只看 pb, pb+1, ..., pb+7 这 8 个物理地址上的数据
      所以 B 要按 [n][k] 转置着放 smem，这样"同一列相邻两行"变成"同一行相邻两列"，
      所以把 B 读进共享内存的时候，B 的访问地址按照列主序来写  BsT[n][k]=B[k*8+n]

    ---- C 的 4 个 float ----
      c0,c1 = C[g  ][t], C[g  ][t+1]
      c2,c3 = C[g+8][t], C[g+8][t+1]

    ---- 为什么 ldmatrix 要 _cvta_generic_to_shared ----
      ldmatrix 的 [addr] 必须是 **shared state space** 的地址（32 位），
      不能直接传 C++ 指针（那是 generic 地址）

    ---- 两个坑 ----
      1) ldmatrix 要求 16 B 对齐：所以每行 8 个 half 的起始偏移必须是 8 的倍数
      2) B 若不做转置，就必须用 .trans 变体，且地址规则不同 —— 这里选转置，省事
*/

#include <cuda_fp16.h>

#define WARP 32

__global__ void mma_warmup_kernel(const half* __restrict__ A,
                                  const half* __restrict__ B,
                                  float* __restrict__ C)
{
    // smem 布局：As 行主序（ld = 16）；BsT 转置后按 [n][k]（ld = 16）
    __shared__ half As[16][16];
    __shared__ half BsT[8][16];

    const int lane = threadIdx.x & 31;

    // ---- 装 smem ----
    // A: 256 个元素 / 32 lane = 每 lane 8 个
    for (int i = lane; i < 16 * 16; i += WARP)
    {
      As[i / 16][i % 16] = A[i];
    }
        

    // B: 输入是 [k][n]（k 走 16、n 走 8），转置成 BsT[n][k]
    // smem 里 As[16][16] 和 BsT[8][16] 都是行主序的二维数组（ld 分别是 16 和 16），只是 BsT 存的是转置后的 B
    // 因为 ldmatrix 加载数据时，要连续的数据，在smem中也要连续，BsT要列连续
    for (int i = lane; i < 8 * 16; i += WARP)
    {
        const int n = i / 16;
        const int k = i % 16;
        BsT[n][k] = B[k * 8 + n];
    }
    __syncwarp();   // 这里只有一个warp，不需要跨warp用__syncthreads()

    // ---- ldmatrix 装 A：x4，一次拿满 4 个寄存器 ----
    // 地址规则：lane 0-7 给矩阵0 的行，8-15 给矩阵1，16-23 给矩阵2，24-31 给矩阵3
    //   矩阵0 = A[0..7][0..7]    矩阵1 = A[8..15][0..7]
    //   矩阵2 = A[0..7][8..15]   矩阵3 = A[8..15][8..15]
    // 所以行 = lane % 16，列组 = (lane / 16) * 8
    /*
      ldmatrix.x4 装的 4 个矩阵：是 4 个 8×8，不是 4 个 16×16
      A 是 16×16 = 256 个 half，x4 装 4 个 8×8 = 4×64 = 256 个 half，正好是全部 A
      命名里面的 m8n8.x4 就是 8*8的块，装4块

      A[16][16] 切成 4 块：
      ┌─────────┬─────────┐
      │ 块0     │ 块2     │   ← 块0: A[0..7][0..7]    块2: A[0..7][8..15]
      │ A[0:8]  │ A[0:8]  │
      │ [0:8]   │ [8:16]  │
      ├─────────┼─────────┤
      │ 块1     │ 块3     │   ← 块1: A[8..15][0..7]   块3: A[8..15][8..15]
      │ A[8:16] │ A[8:16] │
      │ [0:8]   │ [8:16]  │
      └─────────┴─────────┘

    */
   // 下面是 内联PTX汇编，把 地址和4个寄存器 交给ldmatrix指令
   // pa 每条lane给一个 首行地址；只给 一行8个half的首地址，不是整个块，然后每条 lane 用自己的 pa 地址，从 smem 连续读 8 个 half
   // 读完后，硬件把这 256个half 重新排列到 32*4 个寄存器里，按照刚刚设计的布局分发
   // 分发规则是硬件写死的，不用自己写
    const half* pa = &As[lane % 16][(lane / 16) * 8];       
    unsigned a0, a1, a2, a3;      // 4个寄存器 ———— 把 smem 的数据搬到 Tensor Core 能读的地方
    // A 是 16×16 = 256 half  ————> 32 lane 平分 = 8 half/lane  ————> 8 half 打包成 32 位寄存器 = 4 个寄存器
    /*
      asm volatile( "指令模板" : 输出操作数 : 输入操作数 : 破坏列表 ) 内联PTX
      ldmatrix   从smem加载矩阵
      .sync      warp 内同步 
      .aligned   所有 lane 必须执行同一条指令
      .m8n8      每个块是 8 行 × 8 列
      .x4        一次搬 4 个 8×8 块
      .shared    地址在 shared memory 空间
      .b16       每个元素 16 bit（half）
      {%0,%1,%2,%3}    输出操作数列表，4 个 32 位寄存器。%0 对应 a0，%1 对应 a1
      [%4]       输入地址操作数
      CUDA 的指针(pa)是 generic 地址（64 位），可以指向 global/shared/local, 但 ldmatrix 的 [addr] 操作数只认 shared 空间的 32 位偏移
      generic 地址直接传，ldmatrix 会把 64 位地址当 32 位用，地址错、读垃圾
      使用 __cvta_generic_to_shared 把 generic 地址转成 shared 空间的 32 位地址； unsigned 强转成 32位无符号

      约束符：
        约束	          含义	                            用在哪
        "=r"	    输出，32 位通用寄存器	                ldmatrix 的输出
        "r"	      输入，32 位通用寄存器	                地址、A/B 操作数
        "+f"	    读写，32 位浮点	                      mma 的 C（累加）
        "+r"	    读写，32 位通用	                      少见，知道有就行
    */ 
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
                 : "r"((unsigned)__cvta_generic_to_shared(pa)));


    // ---- ldmatrix 装 B：x2 ----
    // BsT 是 [8][16]（[n][k]），K 维是 16 列; ldmatrix 每块只搬 8 列，所以 16 列要 2 块
    // &1 是为了让 lane 16-31 的指针不越界  
    const half* pb = &BsT[lane % 8][((lane / 8) & 1) * 8];
    unsigned b0, b1;
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(b0), "=r"(b1)
                 : "r"((unsigned)__cvta_generic_to_shared(pb)));

    // ---- 一条 mma：C += A · B ----
    // C[16×8] = A[16×16] · B[16×8] + C
    float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;           // 累加器初始值
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));    // a0..a3 是 A 的 4 个寄存器,b0..b1 是 B 的 2 个寄存器

    // ---- 按 C 的布局写回 ----
    // 把 mma.sync 算出的 c0..c3 按 C 的寄存器映射写回全局内存 C[16][8]
    // c0..c3 是 mma.sync 输出的 4 个 fp32 寄存器
    // 每条 lane 只持有 C 的 128 个元素里的 4 个，写回的时候只能写自己那4个位置；32 lane × 4 = 128 = 16×8
    // 行 = lane/4，列 = (lane%4)*2，+8 跳半块（行 +8），+1 跳相邻列
    // C 是 [16][8]、ld=8、c2/c3 行 +8、C 是 float 直接赋值不转 half
    const int g = lane >> 2;            // 行号 0 1 ... 7
    const int t = (lane & 3) * 2;       // 起始列 0 2 4 6
    C[g * 8 + t] = c0;
    C[g * 8 + t + 1] = c1;
    C[(g + 8) * 8 + t] = c2;
    C[(g + 8) * 8 + t + 1] = c3;
}


#include <torch/extension.h>

torch::Tensor mma_warmup(torch::Tensor A, torch::Tensor B)
{
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "A/B must be CUDA tensors");
    TORCH_CHECK(A.dtype() == torch::kHalf && B.dtype() == torch::kHalf, "A/B must be fp16");
    TORCH_CHECK(A.dim() == 2 && A.size(0) == 16 && A.size(1) == 16, "A must be [16,16]");
    TORCH_CHECK(B.dim() == 2 && B.size(0) == 16 && B.size(1) == 8,  "B must be [16,8]");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "A/B must be contiguous");

    auto C = torch::empty({16, 8}, A.options().dtype(torch::kFloat));
    mma_warmup_kernel<<<1, WARP>>>(
        reinterpret_cast<const half*>(A.data_ptr()),
        reinterpret_cast<const half*>(B.data_ptr()),
        C.data_ptr<float>());

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mma_warmup launch failed: ", cudaGetErrorString(err));
    return C;
}
