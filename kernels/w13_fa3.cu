/*
    fa3.cu —— FlashAttention-3 前向（Hopper：TMA + wgmma + warp specialization + 跨 tile 重叠）

    一个 block 管 TILE_M = 128 行 Q，384 线程 = 3 个 warpgroup：
      wg0      producer  —— 只发 TMA，寄存器 40
      wg1/wg2  consumer  —— 只算，寄存器 232；各管 64 行
      （这两个预算要真正生效，必须有 __launch_bounds__(384, 1)；只有单参数时 ptxas 会忽略
        setmaxnreg，见 kernel 声明处的注释。生效后 SASS 里是 USETMAXREG.ALLOC/DEALLOC，
        不是 "SETMAXNREG" —— grep 的时候别写错，否则会数出 0 条）

    同步全靠 mbarrier 对：full[buf]（TMA 到齐，tx 计数）/ empty[buf]（8 个 consumer warp 用完）

    重叠是这一版的核心：发完 QK(jt+1) 再去算 softmax(jt)，让 Tensor Core 与 SFU 同时在跑
    wgmma_wait1 就是那道闸门（允许最新一组继续飞）
    为此要三份 K/V buffer，并且 PV 的 buffer 延迟一轮才归还

    未做：pingpong（两个 consumer 错拍；寄存器预算现在已经真的生效，卡点不在 setmaxnreg 了）、
          对角线块特化、persistent kernel、TMA multicast、FP8

    判据（本地只需编译 + SASS）：
      UTMALDG.3D ≥ 7、HGMMA.64x32x16 ≥ 12、HGMMA.64x64x16 ≥ 4、LDGSTS = 0、USETMAXREG = 2；
      上机后三档 max|diff| ≤ 3e-3（注意：wgmma 版不可能与 4060 上 mma.sync 版的
      3.644e-04 / 4.944e-04 / 5.429e-04 位级一致，判据是 3e-3 那条线）
*/




#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include <cstdio>


#define TILE_M 128              // 一个 block 负责的 Q：3 个 warpgroup = 384 线程，consumer 两个各管 64 行
#define TILE_N 32               // 一次处理多少个 key
#define HEAD_DIM 64
#define SM_THREADS 384          // 3 个 warpgroup：1 producer + 2 consumer
#define WARP_M 16               // 一个 warp 负责 16 行（QK 与 PV 两阶段相同）
#define WWG 64                  // 一个 warpgroup 负责 64 行
#define PROD_WARPS 4            // producer 的 warp 数
#define CONS_WARPS (SM_THREADS / 32 - PROD_WARPS)       // consumer 的 warp 数  8
#define NBUF 3                  // 三份 K/V buffer：producer 要能跑到 consumer 前面两轮
#define NEG_INF (-1e30f)


// ===== mbarrier.arrive（只到、不等）与 setmaxnreg =====
__device__ __forceinline__ void mbar_arrive(uint64_t *bar)
{
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile("mbarrier.arrive.shared.b64 _, [%0];\n" ::"r"(a));
}

__device__ __forceinline__ void wg_reg_dec_producer()
{
    // setmaxnreg.dec 40  主动降到 40，必须是 8 的倍数
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n");
}
__device__ __forceinline__ void wg_reg_inc_consumer()
{
    // setmaxnreg.inc 232 升到 232
    asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n");
}


// ---------------------------------------------------------------------------
// 基础工具
// ---------------------------------------------------------------------------
__device__ __forceinline__ unsigned pack2(float a, float b)
{
    const half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<const unsigned*>(&h);
}

__device__ __forceinline__ float warp_max4(float v)
{
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
    return v;
}
__device__ __forceinline__ float warp_sum4(float v)
{
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    v += __shfl_xor_sync(0xffffffffu, v, 2);
    return v;
}

// wgmma 的 64 位矩阵描述符
__device__ __forceinline__ unsigned long long make_desc(const void* p, unsigned lbo,
                                                        unsigned sbo, unsigned swz_mode = 0)
{
    const unsigned long long addr = (unsigned long long)__cvta_generic_to_shared(p);
    unsigned long long d = (addr >> 4) & 0x3FFFu;
    d |= ((unsigned long long)(lbo >> 4) & 0x3FFFu) << 16;
    d |= ((unsigned long long)(sbo >> 4) & 0x3FFFu) << 32;
    d |= ((unsigned long long)(swz_mode & 0x3u)) << 62;
    return d;
}

__device__ __forceinline__ void wgmma_fence()
{
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void wgmma_commit()
{
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void wgmma_wait0()
{
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
}
// 允许还有 1 组在飞 ———— 这就是"重叠"的闸门
__device__ __forceinline__ void wgmma_wait1()
{
    asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory");
}

__device__ __forceinline__ void wgmma_m64n32k16(float (&sc)[TILE_N / 8][4],
                                                unsigned long long a_desc,
                                                unsigned long long b_desc)
{
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, "
        "%16, %17, 1, 1, 1, 0, 0;\n"
        : "+f"(sc[0][0]), "+f"(sc[0][1]), "+f"(sc[0][2]), "+f"(sc[0][3]), "+f"(sc[1][0]), "+f"(sc[1][1]),
          "+f"(sc[1][2]), "+f"(sc[1][3]), "+f"(sc[2][0]), "+f"(sc[2][1]), "+f"(sc[2][2]), "+f"(sc[2][3]),
          "+f"(sc[3][0]), "+f"(sc[3][1]), "+f"(sc[3][2]), "+f"(sc[3][3])
        : "l"(a_desc), "l"(b_desc));
}


// RS 形式：A 来自寄存器（每线程 4 个 32 位寄存器 = 8 个 half），B 来自 smem 描述符
// 注意：PV 不用它 —— V 是 N-major，RS 的 B 必须 K-major，会读出 smem 边界。见 run_softmax_pv。
__device__ __forceinline__ void wgmma_m64n64k16_rs(float (&d)[HEAD_DIM / 8][4],
                                                   const unsigned (&a)[4],
                                                   unsigned long long b_desc)
{
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, "
        "{%32,%33,%34,%35}, %36, 1, 1, 1, 0;\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]),
          "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
          "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
          "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]),
          "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]),
          "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]),
          "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]),
          "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(b_desc));
}

// mbarrier
__device__ __forceinline__ void mbar_init(uint64_t *bar, uint32_t count)
{
    uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(a), "r"(count));
}

__device__ __forceinline__ void mbar_arrive_expect_tx(uint64_t *bar, uint32_t bytes)
{
    uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(a), "r"(bytes));
}

__device__ __forceinline__ void mbar_wait(uint64_t *bar, uint32_t phase)
{
    uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile(
    "{\n"
    ".reg .pred p;\n"
    "WAIT_%=:\n"
    "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
    "@!p bra WAIT_%=;\n"
    "}\n" ::"r"(a), "r"(phase));
}

// 3D TMA：把 box [HEAD_DIM, TILE_N, 1] 搬到 smem，坐标 (x=D, y=token, z=bh)
__device__ __forceinline__ void tma_load_3d(void *dst, uint64_t *bar,
                                            const CUtensorMap *tmap,
                                            int x, int y, int z)
{
    uint32_t s = (uint32_t)__cvta_generic_to_shared(dst);
    uint32_t b = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%3, %4, %5}], [%2];\n"
        :: "r"(s), "l"(tmap), "r"(b), "r"(x), "r"(y), "r"(z)
        : "memory");
}


// ---------------------------------------------------------------------------
// kernel
// ---------------------------------------------------------------------------
// 第二个参数（每 SM 最少 1 个 block）必须写：只写 __launch_bounds__(384) 时 
// ptxas 算不出入口寄存器数，setmaxnreg 会被直接忽略（ptxas info C7508，SASS 里一条 USETMAXREG 都没有）
// 写成 (384, 1) 后：静态 168 regs/线程（384×168 = 64512 ≤ 65536，正好 1 CTA/SM）
// producer 再 dec 到 40、consumer inc 到 232 —— 8×32×232 + 4×32×40 = 64512 正好塞满
__global__ void __launch_bounds__(SM_THREADS, 1) fa3_kernel(const __grid_constant__ CUtensorMap qtmap,
                                                         const __grid_constant__ CUtensorMap vmap,
                                                         const __grid_constant__ CUtensorMap kmap,
                                                         half*  __restrict__ O,
                                                         int B, int H, int Hkv, int S, float scale)
{
    // ---- smem ----
    // SW128 的 swizzle 相位来自地址 bit 7-9，所以每块基址要按 1024 B 对齐
    // 现在靠 16384 / 28672 这些偏移恰好成立，改声明顺序或加填充就会静默错
    __shared__ alignas(1024) half Qs[TILE_M * HEAD_DIM];            // 16 KB 128B swizzle（Q 也是 TMA 拿的）
    __shared__ alignas(1024) half Kbuf[NBUF][TILE_N * HEAD_DIM];    // 3 × 4 KB 128B swizzle
    __shared__ alignas(1024) half Vbuf[NBUF][TILE_N * HEAD_DIM];    // 3 × 4 KB swizzle

    __shared__ alignas(8) uint64_t full[NBUF];                      // TMA 搬完翻相位 —— consumer 等
    __shared__ alignas(8) uint64_t empty[NBUF];                     // 8 个 consumer warp 各 arrive 一次 —— producer 等
    __shared__ alignas(8) uint64_t qbar;                            // Q 的 TMA 完成计数（count=1）

    const int bx = blockIdx.x;
    const int bh = blockIdx.y;
    const int b = bh / H;
    const int h = bh % H;
    const int kvh = h / (H / Hkv);
    const int kv_bh = b * Hkv + kvh;        // 描述符第三维的下标

    const int row_base = max(0, min(bx * TILE_M, S - TILE_M));

    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int warp_id = tid / 32;
    const int wg_id = warp_id / 4;                  // 0 = producer, 1/2 = consumer
    const int warpg_id = warp_id % 4;
    const bool is_prod = (wg_id == 0);
    const int consumer_id = wg_id - 1;              // 0/1

    const int row0 = consumer_id * WWG + warpg_id * WARP_M + (lane / 4);
    const int row1 = row0 + 8;

    // 初始化
    if (tid == 0)
    {
        for (int i = 0; i < NBUF; ++i)
        {
            mbar_init(&full[i], 1);                 // 3 个 full
            mbar_init(&empty[i], CONS_WARPS);       // 3 个 empty
        }
        mbar_init(&qbar, 1);                        // 1 个 qbar
    }
    __syncthreads();
    // mbarrier.init是“generic proxy”（普通线程）写的 smem
    // TMA 是“async proxy”（异步硬件）读的 smem
    // 两个 proxy 之间的可见性不保证 ———— 必须补 fence.proxy.async
    // 保证“generic proxy 写的 mbarrier 初始化”对“async proxy 的 TMA”可见
    // 两个 proxy 之间必须补栅栏，否则 TMA 可能读到未初始化的 mbarrier，complete_tx 写错地方
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");

    // ---- Q 进 smem ----
    if (tid == 0)
    {
        mbar_arrive_expect_tx(&qbar, TILE_M * HEAD_DIM * sizeof(half));
        tma_load_3d(Qs, &qbar, &qtmap, 0, row_base, bh);
    }
    // 全 block 384 线程都等 Q 到齐
    // 这是唯一一次全 block 同步——之后 producer 和 consumer 按角色分开，靠 mbarrier 同步
    mbar_wait(&qbar, 0);


    const int jt_end = min(row_base + TILE_M - 1, S - 1) / TILE_N;

    // ================= producer：只搬数据，不碰 mma =================
    if (is_prod)
    {
        wg_reg_dec_producer();

        for (int jt = 0; jt <= jt_end; ++jt)
        {
            // 同一块 buffer 每 3 轮用一次
            const int buf = jt % NBUF;      // 0 1 2 0 1 2 
            // jt = 0,1,2 是预填充阶段：三份 empty 的 buffer 都还是空的，不用等
            if (jt >= NBUF)
            {
                // producer 在 jt 轮等的是 jt-3 轮 consumer 翻的 empty
                const int empty_phase = ((jt - NBUF) / NBUF) & 1;
                mbar_wait(&empty[buf], empty_phase);
            }

            if (tid == 0)
            {
                mbar_arrive_expect_tx(&full[buf], 2 * TILE_N * HEAD_DIM * (uint32_t)sizeof(half));
                tma_load_3d(Kbuf[buf], &full[buf], &kmap, 0, jt * TILE_N, kv_bh);
                tma_load_3d(Vbuf[buf], &full[buf], &vmap, 0, jt * TILE_N, kv_bh);
            }
        }
        return;   // producer 到此为止
    }

    // ================= consumer：只算，不搬 =================
    wg_reg_inc_consumer();

    const float qk_scale = scale;
    float m0 = NEG_INF, m1 = NEG_INF;
    float l0 = 0.f, l1 = 0.f;
    float acc[HEAD_DIM / 8][4];
    for (int nd = 0; nd < HEAD_DIM / 8; ++nd)
        for (int c = 0; c < 4; ++c) acc[nd][c] = 0.f;

    // 两份 S 累加器：scA 与 scB 交替
    // QK(jt+1) 写 scB 时，softmax(jt) 正在读 scA，避免 写 QK 时覆盖正在 softmax 的 S
    // 必须写成两个具名数组，若写成 sc[2][..][..] 再用运行时下标 sc[slot] 访问，编译器只会把它放进 local memory（每线程 128 B 栈帧），寄存器版就白写了
    // 下面 slot = jt & 1       偶数为scA，奇数为scB
    float scA[TILE_N / 8][4];
    float scB[TILE_N / 8][4];

    const int jt_end_wg = min(row_base + consumer_id * WWG + WWG - 1, S - 1) / TILE_N;

    // 本 warpgroup 的行指针与描述符参数
    half* qwg = Qs + consumer_id * (WWG * HEAD_DIM);
    const unsigned a_lbo = 1,   a_sbo = 1024;     // A(Q)：128B swizzle
    const unsigned b_lbo = 1,   b_sbo = 1024;     // B(K)：128B swizzle

    // ---- 发 QK(jt),把 HEAD_DIM 按 16 切成 4 段各发一条 wgmma ----
    // s 是要被写的那份累加器（scA 或 scB），由调用点按 jt 的奇偶决定
    auto issue_qk = [&](float (&s)[TILE_N / 8][4], int jt)
    {
        const int buf = jt % NBUF;
        for (int n = 0; n < TILE_N / 8; ++n)
            for (int c = 0; c < 4; ++c) s[n][c] = 0.f;  // 清零累加器，每个tile重新算S

        wgmma_fence();                                  // 清零是写寄存器，wgmma来读，必须fence
        for (int kk = 0; kk < HEAD_DIM / 16; ++kk)
        {
            const unsigned long long ad = make_desc(qwg + kk * 16, a_lbo, a_sbo, 1);
            const unsigned long long bd = make_desc(Kbuf[buf] + kk * 16, b_lbo, b_sbo, 1);
            wgmma_m64n32k16(s, ad, bd);
        }
        wgmma_commit();
    };

    // ---- softmax(jt) + PV(jt) 收进一个 lambda：s 是本 tile 的那份 S 累加器 ----
    auto run_softmax_pv = [&](float (&s)[TILE_N / 8][4], int jt, int buf)
    {
        const int gr0 = row_base + row0;
        const int gr1 = row_base + row1;
        // 必须用运行最大值 m0/m1 播种：这一 tile 对某一行可能整块被因果屏蔽，
        // 此时 tile 内最大值就是 NEG_INF(-1e30)，若从 NEG_INF 起算，
        // a0 = exp(m0 - (-1e30)) = +inf → acc/l 变 inf → 最后 inf*0 = NaN
        float mx0 = m0, mx1 = m1;
        for (int n = 0; n < TILE_N / 8; ++n)
        {
            const int j0 = jt * TILE_N + n * 8 + (lane % 4) * 2;
            for (int c = 0; c < 4; ++c)
            {
                const int j = j0 + (c & 1);
                const int gr = (c < 2) ? gr0 : gr1;
                float v = s[n][c] * qk_scale;
                if (j > gr || j >= S) v = NEG_INF;
                s[n][c] = v;
                if (c < 2) mx0 = fmaxf(mx0, v);
                else       mx1 = fmaxf(mx1, v);
            }
        }
        const float mn0 = warp_max4(mx0);
        const float mn1 = warp_max4(mx1);

        const float a0 = __expf(m0 - mn0);
        const float a1 = __expf(m1 - mn1);

        float s0 = 0.f, s1 = 0.f;
        for (int n = 0; n < TILE_N / 8; ++n)
            for (int c = 0; c < 4; ++c)
            {
                const float p = __expf(s[n][c] - ((c < 2) ? mn0 : mn1));
                s[n][c] = p;
                if (c < 2) s0 += p;
                else       s1 += p;
            }
        l0 = l0 * a0 + warp_sum4(s0);
        l1 = l1 * a1 + warp_sum4(s1);
        m0 = mn0; m1 = mn1;

        for (int nd = 0; nd < HEAD_DIM / 8; ++nd)
        {
            acc[nd][0] *= a0; acc[nd][1] *= a0;
            acc[nd][2] *= a1; acc[nd][3] *= a1;
        }

        // ===== 把 softmax 后的 P 打包成 wgmma RS 的 A 片段 =====
        unsigned pA[2][4];
        for (int ks = 0; ks < 2; ++ks)
        {
            const int n0 = 2 * ks + 0;
            const int n1 = 2 * ks + 1;
            pA[ks][0] = pack2(s[n0][0], s[n0][1]);
            pA[ks][1] = pack2(s[n0][2], s[n0][3]);
            pA[ks][2] = pack2(s[n1][0], s[n1][1]);
            pA[ks][3] = pack2(s[n1][2], s[n1][3]);
        }

        // ---- O += P·V：走 mma.sync，不用 wgmma RS ----
        // RS 形式里 B 必须是 K-major；V 是 [key][dim]、dim 连续，属于 N-major 布局，
        // 描述符会把地址算到 smem 外面（H800 实测 Out-of-range shared address）。
        // mma.sync 的 B 走 ldmatrix.trans，天然吃 "V[key][dim]" 这种布局。
        // 代价：PV 变成同步指令；QK 仍然是 wgmma 异步。
        for (int ks = 0; ks < 2; ++ks)
        {
            for (int nd = 0; nd < HEAD_DIM / 8; ++nd)
            {
                const int vr = ks * 16 + (lane % 8) + ((lane / 8) & 1) * 8;
                const int vcol = nd * 8;
                // V 在 smem 里是 TMA 写出来的 128B swizzle：16B 块号与行号异或
                const half* pv = Vbuf[buf] + (size_t)vr * HEAD_DIM +
                                 ((((vcol >> 3) ^ (vr & 7)) << 3) | (vcol & 7));
                unsigned b0, b1;
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                             : "=r"(b0), "=r"(b1)
                             : "r"((unsigned)__cvta_generic_to_shared(pv)));
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                    : "+f"(acc[nd][0]), "+f"(acc[nd][1]), "+f"(acc[nd][2]), "+f"(acc[nd][3])
                    : "r"(pA[ks][0]), "r"(pA[ks][1]), "r"(pA[ks][2]), "r"(pA[ks][3]),
                      "r"(b0), "r"(b1));
            }
        }
    };

    // 主循环：跨 tile 重叠的核心
    // ---- 预填充：先把 QK(0) 发出去（此时还没 softmax 可做，所以这一步没有重叠）----
    mbar_wait(&full[0 % NBUF], (0 / NBUF) & 1);
    issue_qk(scA, 0);                        // jt=0 用 scA

    for (int jt = 0; jt <= jt_end; ++jt)
    {
        const int buf = jt % NBUF;                      // 当前 tile 的 KV 在第几块 buffer
        const int slot = jt & 1;                        // 偶数用scA、奇数用scB
        const bool has_next = (jt + 1) <= jt_end_wg;    // 是否还有下一个tile要算，最后一块不提前发

        // ---- 先把下一个 tile 的 QK 发出去（这时 TC 开始干活）----
        if (has_next)
        {
            const int buf_n = (jt + 1) % NBUF;      // 下一个 tile 的 buffer
            mbar_wait(&full[buf_n], ((jt + 1) / NBUF) & 1);   // 等 producer 把 K/V(jt+1) 搬完
            if (((jt + 1) & 1) == 0) 
            {
            issue_qk(scA, jt + 1);          // jt+1是偶数，就写scA
            }
            else
            {
                issue_qk(scB, jt + 1);      // jt+1是奇数，就写scB
            }                     
        }

        // ---- 闸门：允许最新那一组（QK(jt+1)）还在飞，只要 QK(jt) 与 PV(jt-1) 完成 ----
        if (has_next) wgmma_wait1();
        else          wgmma_wait0();

        // ---- softmax(jt)：这一步在 SFU 上跑，同时 TC 还在跑 QK(jt+1) ----
        if (jt <= jt_end_wg)
        {
            if (slot == 0) run_softmax_pv(scA, jt, buf);
            else           run_softmax_pv(scB, jt, buf);
        }

        // ---- 延迟一轮归还 buffer：本轮还的是 jt-1 的，它的 PV 已在 闸门 里等到 ----
        if (jt >= 1 && lane == 0)
            mbar_arrive(&empty[(jt - 1) % NBUF]);
    }

    // 收尾：最后一块 buffer 的 PV 还没等到
    wgmma_wait0();
    if (lane == 0 && jt_end >= 0)
        mbar_arrive(&empty[jt_end % NBUF]);

    // ---- 收尾：除 l 写回 ----
    const float inv0 = (l0 > 0.f) ? (1.f / l0) : 0.f;
    const float inv1 = (l1 > 0.f) ? (1.f / l1) : 0.f;
    const size_t o_base = ((size_t)b * H + h) * S * HEAD_DIM;
    for (int nd = 0; nd < HEAD_DIM / 8; ++nd)
    {
        const int col = nd * 8 + (lane % 4) * 2;
        if (row_base + row0 < S)
        {
            O[o_base + (size_t)(row_base + row0) * HEAD_DIM + col + 0] = __float2half(acc[nd][0] * inv0);
            O[o_base + (size_t)(row_base + row0) * HEAD_DIM + col + 1] = __float2half(acc[nd][1] * inv0);
        }
        if (row_base + row1 < S)
        {
            O[o_base + (size_t)(row_base + row1) * HEAD_DIM + col + 0] = __float2half(acc[nd][2] * inv1);
            O[o_base + (size_t)(row_base + row1) * HEAD_DIM + col + 1] = __float2half(acc[nd][3] * inv1);
        }
    }
}

// ---------------------------------------------------------------------------
// host
// ---------------------------------------------------------------------------

// 构建 TMA 描述符，要 3D 的
static int make_tensor_map(CUtensorMap *tmap, void *ptr, int BH, int S, int D, int box_rows)
{
    // 在TMA里面x是连续维，y是跳跃维
    cuuint64_t gdim[3]    = {(cuuint64_t)D, (cuuint64_t)S, (cuuint64_t)BH};
    cuuint64_t gstride[2] = {(cuuint64_t)D * sizeof(half),                 // 行跨度 = 128 B
                             (cuuint64_t)D * sizeof(half) * S};            // head 跨度
    cuuint32_t box[3]     = {(cuuint32_t)D, (cuuint32_t)box_rows, 1};
    cuuint32_t estride[3] = {1, 1, 1};

    CUresult r = cuTensorMapEncodeTiled(
        tmap, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 3,
        ptr, gdim, gstride,
        box, estride,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    if(r != CUDA_SUCCESS)
    {
        const char *msg = "unknow";
        cuGetErrorString(r, &msg);
        printf("cuTensorMapEncodeTiled failed: %s\n", msg);
        return -1;
    }
    return 0;
}

#include <torch/extension.h>
torch::Tensor fa3_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, double scale)
{
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "inputs must be CUDA");
    TORCH_CHECK(q.dtype() == torch::kHalf, "q must be fp16");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "q/k/v must be [B,H,S,D]");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(), "must be contiguous");

    const int B = q.size(0), H = q.size(1), S = q.size(2), D = q.size(3);
    const int Hkv = k.size(1);
    TORCH_CHECK(D == HEAD_DIM, "HEAD_DIM must be 64");
    TORCH_CHECK(H % Hkv == 0, "H must be a multiple of Hkv");
    TORCH_CHECK(S >= 1, "S must be >= 1");
    // KV 长度必须等于 Q 长度：TMA 描述符是用 q 的 S 建的，长度不一致会静默算出错的数。
    TORCH_CHECK(k.size(2) == S && v.size(2) == S, "k/v seq len must equal q seq len");
    TORCH_CHECK(v.size(1) == Hkv, "v head count must equal k head count");
    // grid.y = B*H，超过 65535 会启动失败
    TORCH_CHECK((long long)B * H <= 65535, "B*H exceeds grid.y limit 65535");

    auto o = torch::empty_like(q);
    dim3 grid((S + TILE_M - 1) / TILE_M, B * H);

    CUtensorMap vmap, kmap, qtmap;
    int bh_total = B * Hkv;
    TORCH_CHECK(make_tensor_map(&vmap, (void*)v.data_ptr(), bh_total, S, HEAD_DIM, TILE_N) == 0, "vmap build failed");
    TORCH_CHECK(make_tensor_map(&kmap, (void*)k.data_ptr(), bh_total, S, HEAD_DIM, TILE_N) == 0, "kmap build failed");
    TORCH_CHECK(make_tensor_map(&qtmap, (void*)q.data_ptr(), B * H, S, HEAD_DIM, TILE_M) == 0, "qtmap build failed");

    // 启动前清一次错误码，否则上一次残留的错误会被误判成这次 launch 失败
    (void)cudaGetLastError();
    fa3_kernel<<<grid, SM_THREADS>>>(
        qtmap, vmap, kmap,
        (half*)o.data_ptr(), B, H, Hkv, S, (float)scale);
    const cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "fa3 launch failed: ", cudaGetErrorString(err));
    return o;
}
