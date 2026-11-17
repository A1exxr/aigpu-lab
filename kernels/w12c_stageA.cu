/*
    w12c_mma.cu —— FA2 的 mma.sync 版，**Stage A：只算 S = Q·Kᵀ**

    为什么先只做 QK：
      mma.sync 有三个"必须自己摆对"的东西——A 布局、B 布局、C 布局。
      一次全写完，出错时你分不清是哪一个错了。所以：
        Stage A（本文件）：Q 与 K 走 ldmatrix → 16 条 mma → S 落回全局内存，和 torch 的 Q@Kᵀ 比
        Stage B（下一步）：S 加 mask → 寄存器里做 online softmax → P 直接当 A 喂 PV → 写 O

    与 w12b（wmma 版）的关键差别：
      |            | w12b                     | w12c（本文件）                    |
      | S 存在哪     | 必须 store 到 smem        | sc[] 数组，全在寄存器             |
      | 布局知道吗   | 不知道（fragment 黑盒）    | 知道：c0,c1 = 第 g 行、第 2t/2t+1 列 |
      | 指令         | HMMA（wmma 封装）         | HMMA.16816（mma.sync 直出）+ LDSM  |

    ---- 本文件的分工（一个 block = 64 行 Q，4 个 warp 各 16 行）----
      smem：Qs [64][64] 8 KB | Ks [32][64] 4 KB          （Stage B 再加 Vs 4 KB）
      每个 warp：拿自己的 16 行 Q（4 条 ldmatrix.x4，D=64 → 4 个 k 段）
                 对每个 K/V tile（32 个 key）：
                     4 个 n-tile（每 8 个 key 一个）× 4 个 k 段 → 16 条 mma → sc[4] 各 4 个 float
                 把 sc 按 C 布局写回 S_out

    ---- 全局内存里的 S ----
      S_out 形状 [B, H, S, S]，行主序；S_out[bh][i][j] = Q[i]·K[j]（未缩放、未 mask）
*/

#include <cuda_fp16.h>

#define TILE_M     64          // 一个 block 处理 64 行 Q（4 warp × 16 行）
#define TILE_N     32          // 每次处理 32 个 key
#define HEAD_DIM   64
#define SM_THREADS 128
#define WARP_M     16          // 一个 warp 处理 16 行 Q
#define NWARP      (SM_THREADS / 32)
#define SMEM_BYTES (TILE_M * HEAD_DIM * 2 + TILE_N * HEAD_DIM * 2)

__global__ void __launch_bounds__(SM_THREADS)
qk_mma_kernel(const half* __restrict__ Q,
              const half* __restrict__ K,
              float* __restrict__ S_out,
              int B, int H, int Hkv, int S)
{
    extern __shared__ char smem[];
    half* Qs = (half*)smem;                       // [TILE_M][HEAD_DIM]
    half* Ks = Qs + TILE_M * HEAD_DIM;            // [TILE_N][HEAD_DIM]

    const int bh    = blockIdx.y;
    const int b     = bh / H;
    const int h     = bh - b * H;
    const int grp   = H / Hkv;
    const int kvh   = h / grp;

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;
    const int g       = lane >> 2;                // 行组：0..7（+8 是下一半）
    const int tcol    = (lane & 3) * 2;           // 列：0,2,4,6（+8 跳后半）

    const size_t q_base = ((size_t)b * H   + h  ) * S * HEAD_DIM;
    const size_t k_base = ((size_t)b * Hkv + kvh) * S * HEAD_DIM;
    const int row_base  = blockIdx.x * TILE_M;
    const int qrow      = row_base + warp_id * WARP_M;

    // ---- Q 块进 smem（TILE_M × HEAD_DIM 个 half，128 线程各装 32 个）----
    for (int i = tid; i < TILE_M * HEAD_DIM; i += SM_THREADS)
    {
        const int r = i / HEAD_DIM, d = i - r * HEAD_DIM;
        const int row = row_base + r;
        Qs[i] = (row < S) ? Q[q_base + (size_t)row * HEAD_DIM + d] : __float2half(0.f);
    }
    __syncthreads();

    // ---- 本 warp 的 16 行 Q → 4 个 A 片段（每个 k 段一条 ldmatrix.x4）----
    // 地址规则（与 mma_warmup 完全一致）：行 = lane%16，列组 = (lane/16)*8
    // 第一维 kk，D维有64维，要拆成4个
    // 每个16段都有4个寄存器，一共16个寄存器
    unsigned aq[HEAD_DIM / 16][4];
    for (int kk = 0; kk < HEAD_DIM / 16; ++kk)          // 4个D维段
    {
        // Qs 是扁平 half*：行跨度 HEAD_DIM
        const int qrow_s = warp_id * WARP_M + (lane % 16);      // 本warp的16行中的第几行 0 1 ... 15
        // 每条 lane 的行首地址
        // Qs是[16][64]，kk遍历D维段
        // kk * 16 是D维段偏移(列偏移)，kk 选 D 的 16 维段，跳过 kk * 16 个元素
        // 然后才是D段内部的列偏移  左半 0..7 还是右半 8..15
        const half* p = Qs + qrow_s * HEAD_DIM + kk * 16 + (lane / 16) * 8;     
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                     : "=r"(aq[kk][0]), "=r"(aq[kk][1]), "=r"(aq[kk][2]), "=r"(aq[kk][3])
                     : "r"((unsigned)__cvta_generic_to_shared(p)));
    }

    // ---- 主循环：每个 K tile 32 个 key ----
    // Stage A 不做 causal，全量循环；Stage B 会加 jt_end = row_last / TILE_N
    for (int jt = 0; jt * TILE_N < S; ++jt)
    {
        // K tile 进 smem（越界填 0）
        for (int i = tid; i < TILE_N * HEAD_DIM; i += SM_THREADS)
        {
            const int jj = i / HEAD_DIM, dd = i - jj * HEAD_DIM;
            const int j  = jt * TILE_N + jj;
            // 这里就可以按照行主序访问然后存储
            // 因为ldmatrix要的K就是D维连续的K
            Ks[i] = (j < S) ? K[k_base + (size_t)j * HEAD_DIM + dd] : __float2half(0.f);
        }
        __syncthreads();

        // sc[nt]：第 nt 个 n-tile（8 个 key）的累加器，每个 lane 4 个 float
        float sc[TILE_N / 8][4];
        for (int nt = 0; nt < TILE_N / 8; ++nt)
            for (int c = 0; c < 4; ++c) sc[nt][c] = 0.f;

        // ---- 16 条 mma：4 个 n-tile × 4 个 k 段 ----
        for (int nt = 0; nt < TILE_N / 8; ++nt)
        {
            for (int kk = 0; kk < HEAD_DIM / 16; ++kk)
            {
                // B 片段：lane 拿 K[nt*8 + g][kk*16 + 2t] 与 [.. +8]
                //  -> 行 = nt*8 + lane%8，列组 = (lane/8)&1 * 8
                const int krow_s = nt * 8 + (lane % 8);
                const half* p = Ks + krow_s * HEAD_DIM + kk * 16 + ((lane / 8) & 1) * 8;
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
        __syncthreads();

        // ---- 按 C 布局写回 S_out ----
        // c0,c1 = 第 (qrow+g) 行、第 jt*TILE_N + nt*8 + 2t / +1 列
        // c2,c3 = 第 (qrow+g+8) 行、同样的列
        for (int nt = 0; nt < TILE_N / 8; ++nt)
        {
            const int j0 = jt * TILE_N + nt * 8 + tcol;
            const int r0 = qrow + g;
            const int r1 = qrow + g + 8;
            float* srow0 = S_out + ((size_t)bh * S + r0) * S;
            float* srow1 = S_out + ((size_t)bh * S + r1) * S;
            if (r0 < S)
            {
                if (j0 + 0 < S) srow0[j0 + 0] = sc[nt][0];
                if (j0 + 1 < S) srow0[j0 + 1] = sc[nt][1];
            }
            if (r1 < S)
            {
                if (j0 + 0 < S) srow1[j0 + 0] = sc[nt][2];
                if (j0 + 1 < S) srow1[j0 + 1] = sc[nt][3];
            }
        }
        __syncthreads();
    }
}


#include <torch/extension.h>

torch::Tensor qk_mma(torch::Tensor q, torch::Tensor k)
{
    TORCH_CHECK(q.is_cuda() && k.is_cuda(), "q/k must be CUDA tensors");
    TORCH_CHECK(q.dtype() == torch::kHalf && k.dtype() == torch::kHalf, "q/k must be fp16");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4, "q/k must be [B,H,S,D]");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous(), "q/k must be contiguous");

    const int B = q.size(0), H = q.size(1), S = q.size(2), D = q.size(3);
    const int Hkv = k.size(1);
    TORCH_CHECK(D == HEAD_DIM, "head_dim must be 64");
    TORCH_CHECK(S >= TILE_M, "S must be >= 64 (Tail 版下一步再补)");
    TORCH_CHECK(H % Hkv == 0, "H must be divisible by Hkv");

    auto S_out = torch::empty({B, H, S, S}, q.options().dtype(torch::kFloat));
    dim3 grid((S + TILE_M - 1) / TILE_M, B * H);
    qk_mma_kernel<<<grid, SM_THREADS, SMEM_BYTES>>>(
        reinterpret_cast<const half*>(q.data_ptr()),
        reinterpret_cast<const half*>(k.data_ptr()),
        S_out.data_ptr<float>(), B, H, Hkv, S);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "qk_mma launch failed: ", cudaGetErrorString(err));
    return S_out;
}
