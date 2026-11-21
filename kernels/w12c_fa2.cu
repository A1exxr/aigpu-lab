/*
    w12c_fa2.cu —— FA2 前向（causal + GQA）的 mma.sync 完整版（Stage B）

    Stage A（w12c_mma.cu + w12c_qk.py）已经验证：sc[nt][0..3] 就是 S 的 (g,2t),(g,2t+1),(g+8,2t),(g+8,2t+1)
    它是 QK 的单测：以后改 HEAD_DIM / tile 尺寸 / ldmatrix 地址时先跑它（2 秒出结果），
    能立刻分清「QK 布局错」还是「softmax/PV 错」，不用在本文件里二分
    本文件在它之上补三件事：
        causal mask（j > row 或 j >= S → -inf）
        寄存器内 online softmax（lane 内 4 个数 + 跨 4 条 lane 两次 shuffle）
        P 直接按 A 布局打包 → 16 条 mma 算 P·V → 收尾除 l

    ---- 为什么 P 不用过 smem ----
      sc[nt] 的 C 布局： c0,c1 = P[g][2t], P[g][2t+1]      c2,c3 = P[g+8][...]
      mma 的 A 布局：    a0,a1 = P[g][2t], P[g][2t+1]      a2,a3 = P[g+8][...]

      完全一样，所以 P 的两个 k 段直接打包：
        k 段 0（P 的第 0..15 列）：a0a1←sc[0].c0c1  a2a3←sc[0].c2c3  a4a5←sc[1].c0c1  a6a7←sc[1].c2c3
        k 段 1（P 的第 16..31 列）：同上，用 sc[2]、sc[3]
      → w12b 里那两次 Smem 往返（Ss 存 + Ps 存 + 两次 load_matrix_sync）全部消失

    ---- 每个 warp 的寄存器预算 ----
      aq  4 段 × 4 = 16 | sc  4 × 4 = 16 | acc 8 × 4 = 32 | pA 2 × 4 = 8 | m/l 4  → 约 76，实测 ~96
      smem 只有 Qs 8 KB + Ks 4 KB + Vs 4 KB = 16 KB  → 可放 5~6 个 block，占用率 ~42%（w12b 是 16%）

    ---- 写法约定 ----
      行：g = lane/4（+8 是下半）；列：2t = (lane%4)*2（+8 跳后半）
      m0/l0 服务行 qrow+g；m1/l1 服务行 qrow+g+8
*/

#include <cuda_fp16.h>

#define TILE_M     64
#define TILE_N     32
#define HEAD_DIM   64
#define SM_THREADS 128
#define WARP_M     16
#define NEG_INF    (-1e30f)
#define SMEM_BYTES (TILE_M * HEAD_DIM * 2 + 2 * TILE_N * HEAD_DIM * 2)

// 负责 fp32 → fp16 打包成 32 位寄存器
// __forceinline__：强制内联，不生成函数调用，这个函数被调用上千次
// （pA 的 8 个寄存器 × 每个 warp），必须内联，否则 call/ret 开销比函数体还大
__device__ __forceinline__ unsigned pack2(float a, float b)
{
    // 返回类型 unsigned，作为容器装 2 个 fp16 的位模式
    // __float2half_rn(a)：rn = round to nearest 就近舍入 
    // __halves2half2(h_lo, h_hi) 把两个 __half 打包成 __half2
    const __half2 h = __halves2half2(__float2half_rn(a), __float2half_rn(b));
    // 把 __half2 的 32 位位模式“重新解释”成 unsigned
    // reinterpret_cast<const unsigned*>(&h) 是 const unsigned*，是一个指针
    // 但调用这个函数的对象，要的是值，所以必须解引用 返回const unsigned 值 
    return *reinterpret_cast<const unsigned*>(&h);
}

// 4 条 lane 的 max 归约
__device__ __forceinline__ float warp_max4(float v)
{
    // 每条 lane 只有这一行的 2 个分数（c0,c1）
    // 每条 lane 和 lane ^ 1 交换 v
    // 0↔1, 2↔3, 4↔5, ..., 30↔31
    // xor 1 交换：两边都变成两者的 max
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));

    // 每条 lane 和 lane ^ 2 交换
    // 0↔2, 1↔3, 4↔6, 5↔7, ...
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
    // 此时，每 4 条 lane 的值相同，且是这 4 条的 max
    return v;
}

// 4 条 lane 的 sum 归约
// 归约后的和，4 条 lane 都是同一个值
__device__ __forceinline__ float warp_sum4(float v)
{
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    v += __shfl_xor_sync(0xffffffffu, v, 2);
    return v;
}

// 128B swizzle 重排
// 把索引 c 拆成：
            // 块号 b = c >> 3      （0..7，第几个 16 字节块）
            // 块内偏移 = c & 7     （0..7，块内第几个 half）
// r*64 + (b ^ (r&7))*8 + (c&7)
// 传入的 r 和 c 必须是相对于当前 smem tile 的局部坐标，且 c 的单位必须是 half（元素个数），绝不能是字节偏移
__device__ __forceinline__ int swz(int r, int c) 
{
    // x << n  ≡  x * 2^n  ————> << 3 就是成 2的3次方
    return r * HEAD_DIM + ((((c >> 3) ^ (r & 7)) << 3) | (c & 7));          // c & 7 用 | 还是 + 都对
}


__global__ void __launch_bounds__(SM_THREADS) fa2_mma_kernel(const half* __restrict__ Q,
                                                            const half* __restrict__ K,
                                                            const half* __restrict__ V,
                                                            half* __restrict__ O,
                                                            int B, int H, int Hkv, int S, float scale)
{
    extern __shared__ char smem[];
    half* Qs = (half*)smem;                     // [TILE_M][HEAD_DIM]
    half* Ks = Qs + TILE_M * HEAD_DIM;          // [TILE_N][HEAD_DIM]
    half* Vs = Ks + TILE_N * HEAD_DIM;          // [TILE_N][HEAD_DIM]

    const int bh   = blockIdx.y;
    const int b    = bh / H;
    const int h    = bh - b * H;
    const int grp  = H / Hkv;
    const int kvh  = h / grp;

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;
    const int g       = lane >> 2;              // 行号 0 ... 7
    const int tcol    = (lane & 3) * 2;         // 起始列 0 2 4 6

    // 这里是 ldmatrix.x2 只需要 16 个地址，指令硬件在设计上就只从 lane 0-15 读取地址
    // 只用 lane 0-15 给地址，是 .x2 的硬件规则，和 A/B 布局无关
    // lane 0-7 给左半（偏移 0），lane 8-15 给右半（偏移 8），lane 16-31 夹回 0/8（硬件不看，但指针不能越界)
    const int gmask   = ((lane / 8) & 1) * 8;   // ldmatrix 的列组偏移

    // S >= TILE_M 时把块起点夹到 S-TILE_M，保证块内 64 行全部真实存在
    const int row_base = min(blockIdx.x * TILE_M, S - TILE_M);
    const int qrow     = row_base + warp_id * WARP_M;

    const size_t q_base = ((size_t)b * H   + h  ) * S * HEAD_DIM;
    const size_t k_base = ((size_t)b * Hkv + kvh) * S * HEAD_DIM;

    // ---- Q 进 smem ----
    /*
        Qs形状是[64][64]，每行 64个half = 128字节 = 32 words = 32 bank
        每行的起始地址都是128字节的倍数，所以每行都从 bank0 开始
        ldmatrix 是一个 warp 级指令，32 条 lane 都会参与，每个线程都要读 8个half = 16字节
        但硬件不会让 32条lane 在同一个 transaction 里同时发 32个16字节=512字节 的请求，一个shared memory transaction最多处理128字节，所以在16字节粒度下分成4个phase
        所以ldmatrix的8条lane(1个phase，刚好访问16*8=128字节)在加载数据的时候，8条lane各读一行，同一列c
        (128r+2c)/4%32 = (32r+1/2)%32 = (1/2c) % 32
        8条lane全落在同一个bank(c/2)  为 8-way conflict

        通过 swizzle 把每行的 16字节块错开     一条lane的访问粒度就是8*2=16字节，16 字节也是 ldmatrix 的最小访问单位
        swizzle 不改变数组大小，改变元素在smem的存放顺序
        HEAD_DIM = 64 half = 128 字节 = 8 个 16 字节块
        把索引 c 拆成：
            块号 b = c >> 3      （0..7，第几个 16 字节块）
            块内偏移 = c & 7     （0..7，块内第几个 half）
        原布局：Qs[r][c] 存储地址       r*64 + c

        swizzle 后：Qs[r][c] 存储地址   r*64 + (b ^ (r&7))*8 + (c&7)
            (b ^ (r&7))*8：算出打乱后的块起点， c & 7为块内偏移
            ^ 是 异或运算(二进制位相同得 0，不同得 1)，算出新的块号；这里行号r必须参与计算，才能让不同行的数据在bank上错开
                3 ^ 1：
                    011       (3)
                    ^ 001       (1)
                    -------
                    010  -> 十进制是 2
                
                5 ^ 6：
                    101  (5)
                    ^ 110  (6)
                    -------
                    011  -> 十进制是 3

            块号 b 和 r&7 异或 ———— 把每行的 8 个块“循环移位”，每行错开的量不同
            b ^ (r & 7) 能保证不重复，是因为 固定 b，b ^ x 对 x = 0..7 是 0..7 的一个排列
            r & 7 不是随便写的，因为一行是8个块

        ldmatrix 每条 lane 要连续读 8 个 half（16 字节 = 1 个块），swizzle 只换 块的位置，不换 块内 8 个 half 的顺序
        swizzle 要保证的是：每一组 8 条 lane 的 8 个块号互不相同。因为每个 16 字节块占 4 个 bank，8 个不同块就覆盖 32 个 bank，于是这一组内 0 冲突
        剩下 3 组 lane 8–15、16–23、24–31 结构相同，所以也各自 0 冲突

        写 smem 时：Qs[swz(r, c)] = Q[...]，一个 8-half 块仍连续写
        读 smem 时：&Qs[swz(r, c)]，一个 8-half 块仍连续读
        块内顺序没变，ldmatrix 读到的 8 个 half 顺序正确
    */ 
    
    for (int i = tid; i < TILE_M * HEAD_DIM; i += SM_THREADS)
    {
        const int r = i / HEAD_DIM;
        const int d = i - r * HEAD_DIM;
        // Qs[i] = Q[q_base + (size_t)(row_base + r) * HEAD_DIM + d];           // 就不能这样写了，要用swizzle方法打乱
        Qs[swz(r, d)] = Q[q_base + (size_t)(row_base + r) * HEAD_DIM + d];
    }
    __syncthreads();

    // ---- Q → 4 个 A 片段 ----
    // 忘记了就看stageA的注释
    unsigned aq[HEAD_DIM / 16][4];              // Q[16][64]   4个D维段，每段都有4个寄存器
    for (int kk = 0; kk < HEAD_DIM / 16; ++kk)  // 遍历4个D维段
    {
        // 行地址 = lane % 16   
        // lane0-7 和 lane16-23 提供的行地址一样，但是K方向列段不一样 k+0...7 和 k+8...15
        const int row_swz_q = warp_id * WARP_M + (lane % 16);       // 本warp的16行中的第几行 0 1 ... 15

        // 列组 = lane / 16      0 或 1，选左半（列 0..7）或右半（列 8..15）
        // ldmatrix.x4 要搬的 A 是16*16，被切成 4 个 8×8 块，所以D段内部也要分 左右8个
        const int col_swz_q = kk * 16 + (lane / 16) * 8;            
        // const half* p = Qs + (size_t)(warp_id * WARP_M + (lane % 16)) * HEAD_DIM + kk * 16 + (lane / 16) * 8;        // 换swizzle方法读取smem
        const half* p = Qs + swz(row_swz_q, col_swz_q);
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                     : "=r"(aq[kk][0]), "=r"(aq[kk][1]), "=r"(aq[kk][2]), "=r"(aq[kk][3])
                     : "r"((unsigned)__cvta_generic_to_shared(p)));
    }

    // ---- online softmax 状态 + O 累加器（全在寄存器）----
    /*
        C 布局里 c0 c1是上半块行 g，c2 c3是下半块行 g+8，同一lane负责两行，所以两套状态
        acc[8][4] 是 O = P·V 的输出累加器
        O_warp[16][64] = P_warp[16][32] · V[32][64]
        mma 是 m16n8k16，N=8，所以 64 列拆成 64/8 = 8 个 n-tile
        每个 n-tile 输出[16][8]，每条 lane 拿 16 * 8 / 32 = 4 个 float(n-tile 里的)
        所以 形状是 acc[8][4]，里面放的值是：每个 n-tile 里，本 lane 的 4 个 float
    */ 
    float m0 = NEG_INF, m1 = NEG_INF, l0 = 0.f, l1 = 0.f;
    float acc[HEAD_DIM / 8][4];
    for (int nd = 0; nd < HEAD_DIM / 8; ++nd)
        for (int c = 0; c < 4; ++c) 
        {
            acc[nd][c] = 0.f;
        }

    const int row0  = qrow + g;
    const int row1  = qrow + g + 8;
    const int jt_end = min(row_base + TILE_M - 1, S - 1) / TILE_N;

    for (int jt = 0; jt <= jt_end; ++jt)
    {
        // K/V tile 进 smem（越界填 0，靠 mask 挡）
        for (int i = tid; i < TILE_N * HEAD_DIM; i += SM_THREADS)
        {
            const int jj = i / HEAD_DIM;
            const int dd = i - jj * HEAD_DIM;
            const int j  = jt * TILE_N + jj;
            const bool ok = (j < S);
            // Ks和Vs D维都是连续的
            // Ks[i] = ok ? K[k_base + (size_t)j * HEAD_DIM + dd] : __float2half(0.f);     // Ks和Vs也有bank conflict，换成swizzle
            // Vs[i] = ok ? V[k_base + (size_t)j * HEAD_DIM + dd] : __float2half(0.f);
            Ks[swz(jj, dd)] = ok ? K[k_base + (size_t)j * HEAD_DIM + dd] : __float2half(0.f); 
            Vs[swz(jj, dd)] = ok ? V[k_base + (size_t)j * HEAD_DIM + dd] : __float2half(0.f);
        }
        __syncthreads();

        // S = Q·Kᵀ：4 个 n-tile × 4 个 k 段
        /*
            这里 sc 和 acc 是两次不同 mma 的 C，两次的 N 维不一样
            第一次：S[16][32] = Q[16][64] · Kᵀ[64][32]
            sc：S = Q·Kᵀ，N = key 数 = TILE_N = 32 → 32/8 = 4 个 n-tile
            这里 Q 形状是[16][64] 因为mma.sync是warp级别的指令，一个warp就负责Q的16行，整个Q块是64行，4个warp一起处理


            第二次：O[16][64] = P[16][32] · V[32][64]
            acc：O = P·V，N = D 维 = HEAD_DIM = 64 → 64/8 = 8 个 n-tile

            不管 N 是 32还是64，每个 n-tile 都是 16*8，32条lane平分 = 4
            所以这里 sc形状是[4][4]

            n-tile就是输出的尺寸，指令规定了就是16*8
        */
       // 本lane的4 * 4 = 16 个值
        float sc[TILE_N / 8][4];
        for (int nt = 0; nt < TILE_N / 8; ++nt)
            for (int c = 0; c < 4; ++c) 
            {
                sc[nt][c] = 0.f;
            }
        // 4 个 n-tile 段 x 4 个 K 段
        // 这里算S[16][32] = Q[16][64] · Kᵀ[64][32]
        // 先遍历 输出为 N
        // 内部再遍历 归约维 K
        for (int nt = 0; nt < TILE_N / 8; ++nt)
        {
            for (int kk = 0; kk < HEAD_DIM / 16; ++kk)
            {
                // B的布局，而且是.x2，只有 lane 0-15 读取地址；且 B 的 16×8 切成 2 个块：是 8 行 × 2 个 K 段
                // 16 条 lane 覆盖的是“8 行 × 2 个块”
                // 8*8只有8个物理行，所以是 lane % 8 确定在哪一行
                const int row_swz_k = nt * 8 + (lane % 8);
                const int col_szw_k = kk * 16 + gmask;          // gmask 就加到 归约的维度上就行 这里就是K维
                const half* p = Ks + swz(row_swz_k, col_szw_k);
                // const half* p = Ks + (size_t)(nt * 8 + (lane % 8)) * HEAD_DIM + kk * 16 + gmask;        // 读ks要用swizzle读
                unsigned b0, b1;
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                             : "=r"(b0), "=r"(b1)
                             : "r"((unsigned)__cvta_generic_to_shared(p)));
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                    : "+f"(sc[nt][0]), "+f"(sc[nt][1]), "+f"(sc[nt][2]), "+f"(sc[nt][3])
                    : "r"(aq[kk][0]), "r"(aq[kk][1]), "r"(aq[kk][2]), "r"(aq[kk][3]),
                      "r"(b0), "r"(b1));
            }
        }

        // mask + 行内 max（先在 lane 内，再跨 4 条 lane 合并）
        float mx0 = m0, mx1 = m1;
        for (int nt = 0; nt < TILE_N / 8; ++nt)
        {
            // 本lane的起始全局列(key位置)
            const int j0 = jt * TILE_N + nt * 8 + tcol; 
            for (int c = 0; c < 4; ++c)
            {
                // c=0 2时 j = j0； c=1 3时，j = j0 + 1 
                // C布局，c0 c1是相邻列
                const int j  = j0 + (c & 1);        
                // c0,c1 属于上半块行 row0，c2,c3 属于下半块行 row1
                const int rr = (c < 2) ? row0 : row1;
                float v = sc[nt][c] * scale;
                if (j > rr || j >= S) v = NEG_INF;
                sc[nt][c] = v;
                if (c < 2)
                {
                    mx0 = fmaxf(mx0, v);
                }
                else
                {
                    mx1 = fmaxf(mx1, v);
                }
                /*
                    32 lane 覆盖 16×8
                             列:  0    1    2    3    4    5    6    7
                        ┌────┬────┬────┬────┬────┬────┬────┬────┐
                    行 0│L0c0│L0c1│L1c0│L1c1│L2c0│L2c1│L3c0│L3c1│  ← lane 0-3 的 c0,c1
                    行 1│L4c0│L4c1│L5c0│L5c1│L6c0│L6c1│L7c0│L7c1│  ← lane 4-7 的 c0,c1
                    行 2│L8c0│L8c1│...                             ← lane 8-11
                    行 3│...                                       ← lane 12-15
                    行 4│...                                       ← lane 16-19
                    行 5│...                                       ← lane 20-23
                    行 6│...                                       ← lane 24-27
                    行 7│L28c0│...                                 ← lane 28-31
                        ├────┼────┼────┼────┼────┼────┼────┼────┤
                  行 8  │L0c2│L0c3│L1c2│L1c3│L2c2│L2c3│L3c2│L3c3│  ← lane 0-3 的 c2,c3
                  行 9  │L4c2│L4c3│...                             ← lane 4-7 的 c2,c3
                  行 10 │...                                       ← lane 8-11
                  行 11 │...                                       ← lane 12-15
                  行 12 │...                                       ← lane 16-19
                  行 13 │...                                       ← lane 20-23
                  行 14 │...                                       ← lane 24-27
                  行 15 │L28c2│...                                 ← lane 28-31
                        └────┴────┴────┴────┴────┴────┴────┴────┘
                     
                    lane 0-3 覆盖 行 0（上半块）和行 8（下半块），各 8 列
                    lane 4-7 覆盖 行 1 和行 9；

                 sc[nt][0..3] 就是 S 的 c0..c3
                 c < 2（c0,c1）属于行 row0 = qrow + g，就是上面行0~7 
                 c >= 2（c2,c3）属于行 row1 = qrow + g + 8，上面行8~15

                3层求max，第一层就是lane内 c0 c1 比 mx0 (row0的局部)
                                            c2 c3 比 mx1(row1的局部)
                    第二层是 nt 循环累积 4 个 n-tile(每个lane在每个n-tile里只拿两个key)，mx0 就覆盖 row0 的 8 个分数，mx1也是
                    第三层是 warp_max4 合并 4 条lane，mn0 就是 row0 的 32列完整的 max；mn1 是 row1 的完整 max
                    16 行之间不需要再比 max，因为 softmax 按行独立
                
                */
            }
        }
        // 这时候第二层已经结束了，mx0 此时就是 4 个 n-tile 里 本lane负责的 8个分数(4 个 nt × 2 个 c0,c1)的最大值
        // 4 条 lane 一比 = 这一行的全局 max，这一行是 当前 K/V tile 内的一行
        const float mn0 = warp_max4(mx0);
        const float mn1 = warp_max4(mx1);

        const float a0 = __expf(m0 - mn0);       // 首轮 m0=-inf → 0
        const float a1 = __expf(m1 - mn1);
        float s0 = 0.f, s1 = 0.f;
        for (int nt = 0; nt < TILE_N / 8; ++nt)
            for (int c = 0; c < 4; ++c)
            {
                const float p = __expf(sc[nt][c] - ((c < 2) ? mn0 : mn1));
                sc[nt][c] = p;
                if (c < 2) s0 += p;
                else       s1 += p;
            }
        l0 = l0 * a0 + warp_sum4(s0);
        l1 = l1 * a1 + warp_sum4(s1);
        // mn0 是“当前 tile 的 32 个 + 历史 m0”的 max，不是“整行 S 个的 max” ———— 整行的 max 要等所有 jt 循环结束，m0 累积完才是
        m0 = mn0; m1 = mn1;

        // O 累加器按 alpha 缩放（寄存器里，32 次乘法）
        for (int nd = 0; nd < HEAD_DIM / 8; ++nd)
        {
            // c0,c1 属于上半块行 乘 a0
            acc[nd][0] *= a0; acc[nd][1] *= a0;
            acc[nd][2] *= a1; acc[nd][3] *= a1;
        }

        // sc 是 S = Q·Kᵀ 的 C 布局寄存器 经过 mask/softmax 后原地变成 P，而 P 的 C 布局恰好等于下一次 mma 的 A 布局
        // 所以 pack2 直接把 fp32 的 p 转成 fp16 打包成 pA，喂给 PV 的 mma，全程不碰 smem
        // P 是 C 布局，C布局=A布局，c0,c1 对应 a0,a1    c2,c3 对应 a2,a3
        // 所以 P 的两个 C 布局 n-tile（sc[0]、sc[1]）可以直接打包成 A 的两个寄存器

        // sc最后装的是softmax之后的p分数，但是也不能直接算，因为p的类型是fp32，mma 的 A操作数要 fp16 
        // pact2 做两件事：fp32 → fp16 转换； 两个 fp16 打包成 1 个 32 位寄存器
        // m16n8k16 的 A 和 C，用的是同一套“2×2 块”lane 映射
        // A 的 a0,a1（行 g，列 2t, 2t+1）和 C 的 c0,c1（行 g，列 2t, 2t+1）是同一个位置
        // A 的 a2,a3（行 g+8，列 2t, 2t+1）和 C 的 c2,c3（行 g+8，列 2t, 2t+1）是同一个位置

        // P 的 K维 是 TILE_N = 32，mma 的 K=16，所以拆 2 段
        // 把 sc 里 4 个 n-tile（每个 8 个 key）两两拼成 ks 的 16 个 key
        unsigned pA[2][4];
        for (int ks = 0; ks < 2; ++ks)
        {
            /*
                A布局：4个寄存器装8个half
                    a0,a1 = A[g][2t],   A[g][2t+1]        ← 行 g，前 8 列
                    a2,a3 = A[g+8][2t], A[g+8][2t+1]      ← 行 g+8，前 8 列
                    a4,a5 = A[g][2t+8], A[g][2t+9]        ← 行 g，后 8 列
                    a6,a7 = A[g+8][2t+8], A[g+8][2t+9]    ← 行 g+8，后 8 列
                
                C布局：sc[nt][c]；
                nt（0..3）：第几个 n-tile。一个 n-tile = 8 个 key，4 个 n-tile 覆盖 32 个 key；
                c（0..3）：C 布局的 4 个 float，对应 mma 输出的 4 个位置
                    c	            行	        列（key）
                    c = 0	    g（上半块）	    nt*8 + 2t               nt*8是n-tile的全局列偏移
                    c = 1	    g（上半块）	    nt*8 + 2t + 1
                    c = 2	    g+8（下半块）	nt*8 + 2t
                    c = 3	    g+8（下半块）	nt*8 + 2t + 1

                    sc[nt][0] = C[g][nt*8 + 2t]
                    sc[nt][1] = C[g][nt*8 + 2t+1]
                    sc[nt][2] = C[g+8][nt*8 + 2t]
                    sc[nt][3] = C[g+8][nt*8 + 2t+1]


                A 布局（pA[ks][0..3] = a0..a7）
                    pA[ks][0] 要 a0,a1 = A[g][2t], A[g][2t+1]——就是“行 g 的前 8 列”
                    sc[2ks+0][0] 和 sc[2ks+0][1] 正好是 S[g][2t], S[g][2t+1]
            
            */
            pA[ks][0] = pack2(sc[2 * ks + 0][0], sc[2 * ks + 0][1]);
            pA[ks][1] = pack2(sc[2 * ks + 0][2], sc[2 * ks + 0][3]);
            pA[ks][2] = pack2(sc[2 * ks + 1][0], sc[2 * ks + 1][1]);
            pA[ks][3] = pack2(sc[2 * ks + 1][2], sc[2 * ks + 1][3]);
        }

        // O += P·V：2 个 k 段（各 16 个 key）× 8 个 n-tile（每个 8 维）
        for (int ks = 0; ks < 2; ++ks)
        {
            for (int nd = 0; nd < HEAD_DIM / 8; ++nd)
            {
                /*
                    关于转置 .trans
                    V 是 [key][dim]；mma 要 B 操作数要 V 的归约维(K维，这里就是key/token维度)在寄存器里连续
                    用 .trans 从物理行读出数据后，在寄存器内转置，使 key 维变成连续的 Col-Major 布局，满足 MMA 要求

                    Vs 整个数组是连续存储的，D维连续，但 mma 要的同一 D 维的相邻 token这 8 个元素
                    V[t][g]      地址 = t*64 + g
                    V[t+1][g]    地址 = (t+1)*64 + g = t*64 + g + 64, 地址差64，不连续
                    mma 的 B 操作数布局固定是 b0 = {B[t][g], B[t+1][g]}（K 维相邻、N 维相同），
                    而 P·V 里 V 的 K 维是 token、N 维是 D 维

                    .trans仍然从 &Vs[t][d] 连续读 8 个（V[t][d..d+7]，同一 token 的 8 个 D 维），但分发时硬件转置
                    读入（连续）：V[t][d], V[t][d+1], ..., V[t][d+7]    ← 同一 token 的 8 个 D 维
                    分发（转置）：V[t][d], V[t+1][d], ..., V[t+7][d]    ← 同一 D 维的 8 个 token

                    总结：
                        S = Q @ K^T，K形状是[token][dim] (K*N)，Ks是正常存储的，取地址的时候D维是连续的，所以mma.sync在读的时候连续的读8个half
                        但计算的时候 B操作数 要的是K^T[dim][token] 的dim维度连续，然后刚好存进去的Ks就是D维连续的，所以不用trans

                        S = Q @ K^T，K形状是[token][dim]   这里B操作数在读的时候 K本来就dim连续，所以正常读，然后计算的时候算的是Q@K^T而不是Q@K，
                        所以不用对K^T进行.trans，因为K^T的归约维，也就是dim维，本来就是连续的

                        O = P @ V，V形状是K*N，存进去的时候也是正常存，D维连续，但B操作数要的是V的第一维，也就是K维连续，但K维不连续，所以用了.trans


                    关于偏移量
                    偏移量 ((lane / 8) & 1) * 8
                    偏移加在行还是列，不由 .trans 决定，而由  MMA 的 B布局的矩阵的 K 维在物理内存中对应哪个轴  决定  
                    
                    S = Q @ K^T
                        MMA 中 B 的含义：K 的一个 tile [K=16, N=8]，K 维 = dim
                        B 的 K=16 对应物理上的 dim 0..15（同一 token 的 16 个维度）
                        B 的 N=8 对应物理上的  token 0..7（8 个不同的 key）
                        物理内存 Ks[token][dim]：dim 是列
                        ldmatrix.x2 把 16 行的数据分成两个 8×8 块交给硬件；这两个块必须覆盖 MMA 所需的 K=16 的全部范围
                        要覆盖 dim 0..15 ——> 两个块必须在列方向上各占 8
                        所以 lane 8-15 的地址要在列上加 8（gmask）


                    O = P @ V
                        MMA 中 B 的含义：V 的一个 tile [K=16, N=8]，K 维 = token
                        B 的 K=16 对应物理上的 token 0..15（同一 dim 的 16 个时间步）
                        B 的 N=8 对应物理上的  dim 0..7（8 个不同的维度）
                        物理内存 Vs[token][dim]：token 是行
                        要覆盖 token 0..15 → 两个块必须在行方向上各占 8
                        所以 Lane 8-15 的地址要在行上加 8
                    
                    偏移量永远加在归约维（K 维）上
                */
                
                
                const int vr = ks * 16 + (lane % 8) + ((lane / 8) & 1) * 8;         
                const int col_swz__v = nd * 8;
                const half* p = Vs + swz(vr, col_swz__v);
                // const half* p = Vs + (size_t)vr * HEAD_DIM + nd * 8;
                unsigned b0, b1;
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                             : "=r"(b0), "=r"(b1)
                             : "r"((unsigned)__cvta_generic_to_shared(p)));
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                    : "+f"(acc[nd][0]), "+f"(acc[nd][1]), "+f"(acc[nd][2]), "+f"(acc[nd][3])
                    : "r"(pA[ks][0]), "r"(pA[ks][1]), "r"(pA[ks][2]), "r"(pA[ks][3]),
                      "r"(b0), "r"(b1));
            }
        }
        __syncthreads();
    }

    // ---- 收尾：只除一次 l ----
    const float inv0 = (l0 > 0.f) ? (1.f / l0) : 0.f;           // l0 是上半块行 row0 的 softmax 分母
    const float inv1 = (l1 > 0.f) ? (1.f / l1) : 0.f;           // l1 是下半块行 row1 的分母
    for (int nd = 0; nd < HEAD_DIM / 8; ++nd)                   // 8 个 D 维段； nd = 0 → D[0..7]     nd = 1 → D[8..15]
    {
        // row0 * HEAD_DIM：上半块行 row0 的偏移
        // nd * 8：第 nd 个 8 列段的起始
        // tcol：本 lane 的列偏移（(lane%4)*2，0/2/4/6）
        half* o0 = O + q_base + (size_t)row0 * HEAD_DIM + nd * 8 + tcol;
        // acc[8][4] 的 8 就是这 8 个 n-tile，nd 循环逐个写回
        o0[0] = __float2half(acc[nd][0] * inv0);            // o0是指针，o0[0] / o0[1] 是它指向的地址和下一个地址
        o0[1] = __float2half(acc[nd][1] * inv0);            // o0[0] 和 o0[1] 是 同一行、相邻两列，地址差 1；每条 lane 只写相邻两列
        if (row1 < S)
        {
            half* o1 = O + q_base + (size_t)row1 * HEAD_DIM + nd * 8 + tcol;
            o1[0] = __float2half(acc[nd][2] * inv1);
            o1[1] = __float2half(acc[nd][3] * inv1);
        }
    }
}


#include <torch/extension.h>

torch::Tensor fa2_mma_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, double scale)
{
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "q/k/v must be CUDA tensors");
    TORCH_CHECK(q.dtype() == torch::kHalf && k.dtype() == torch::kHalf && v.dtype() == torch::kHalf,
                "q/k/v must be fp16");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "q/k/v must be [B,H,S,D]");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
                "q/k/v must be contiguous");

    const int B = q.size(0), H = q.size(1), S = q.size(2), D = q.size(3);
    const int Hkv = k.size(1);
    TORCH_CHECK(D == HEAD_DIM, "head_dim must be 64");
    TORCH_CHECK(S >= TILE_M, "S must be >= 64");
    TORCH_CHECK(H % Hkv == 0, "H must be divisible by Hkv");

    auto out = torch::empty_like(q);
    dim3 grid((S + TILE_M - 1) / TILE_M, B * H);
    fa2_mma_kernel<<<grid, SM_THREADS, SMEM_BYTES>>>(
        reinterpret_cast<const half*>(q.data_ptr()),
        reinterpret_cast<const half*>(k.data_ptr()),
        reinterpret_cast<const half*>(v.data_ptr()),
        reinterpret_cast<half*>(out.data_ptr()),
        B, H, Hkv, S, (float)scale);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "fa2_mma launch failed: ", cudaGetErrorString(err));
    return out;
}
