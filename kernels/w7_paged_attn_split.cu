// w7 split-L 版：把每个 (b,h) 的 L 次迭代切成 NSPLIT 段，每段一个 warp 独立算，
// 再用 merge_kernel 按 online softmax 规则把 NSPLIT 份部分结果合成最终输出。
//
// 与 w7_paged_attn.cu（单链 2 路展开，1.5084 ms）的唯一区别：
//   1) 工作单元从 B*H=256 个变成 B*H*NSPLIT=1024 个
//   2) 每个 warp 只扫自己那一段 [t_begin, t_end)
//   3) 结尾不写 out，而是把 (m_i, d_i, acc_i) 写进 partial buffer
// 其余（地址计算、2 路展开、warp 归约）与旧版完全一样
#include <torch/extension.h>
#include <cuda_fp16.h>

#define BS      16     // 每块 token 数
#define DMAX    128    // 每线程最多负责 DMAX/32 = 4 个元素
#define NSPLIT  4      // 切几段

// ─────────────────────────────────────────────────────────────
// kernel 1：分段扫描，产出局部 (m, d, acc)
// ─────────────────────────────────────────────────────────────
__global__ void paged_attn_split_kernel(const half* __restrict__ q,
                                        const half* __restrict__ k_cache,
                                        const half* __restrict__ v_cache,
                                        const int*  __restrict__ block_table,
                                        const int*  __restrict__ context_lens,
                                        float* __restrict__ partial_acc,   // [NSPLIT,B,H,D] fp32
                                        float* __restrict__ partial_m,     // [NSPLIT,B,H]   fp32
                                        float* __restrict__ partial_d,     // [NSPLIT,B,H]   fp32
                                        int B, int H, int D, int MAXB)
{
    int gwarp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;   // 全局第几个 warp，0 1.. 1023
    int lane  = threadIdx.x & 31;

    if (gwarp >= B * H * NSPLIT)
    {
        return;
    }

    // ===== TODO 1：把 gwarp 拆成 (b, h, split) =====
    // 约定：同一个 (b,h) 的 NSPLIT 个 split 在 gwarp 上连续编号，
    // 一个(b,h)的D维度分成4段，1个warp负责1段
    // split 只是把同一个 bh 复制成 4 个 gwarp，不改变 bh 的编号
    int bh    = gwarp / NSPLIT;             // 第几个 (b,h)         0 1 2 ... 255 和单链版的gwarp范围一样
    int split = gwarp % NSPLIT;             // 这个 (b,h) 的第几段   0 1 2 3
    int b = bh / H;                         // 第几个请求
    int h = bh % H;                         // 第几个头
    int L = context_lens[b];

    // ===== TODO 2：算这一段负责的 token 区间 [t_begin, t_end) =====
    // chunk = 上取整(L / NSPLIT)；最后一段用 min(L, ...) 夹住，防止越界
    int chunk   = (L + NSPLIT - 1) / NSPLIT;
    int t_begin = split * chunk;                    // 每个warp的t_begin不同，通过split来区分
    int t_end   = min(L, t_begin + chunk);          // 最后一段可能不满

    float acc[4] = {0.f};
    float denom  = 0.f;
    float m      = -INFINITY;

    const half* qrow = q + (size_t)(b * H * D + h * D);

    // ===== TODO 3：主循环，把原来的 for (t = 0; t + 1 < L; t += 2) 改成
    //               只在 [t_begin, t_end) 里做 2 路展开 =====
    // 注意：段内元素个数可能为奇数，收尾块也必须相对 t_end 判断，不要再用 L。
    int t = t_begin;
    // 现在的for循环扫完这一段，只是[t_begin,t_end]这一段的token，得到这一段的3个值：m、denom、acc[4]
    // 一个warp算的一段就是一个partial，4个partial合起来才是完整结果
    // merge kernel就是把这4个partial合成1个
    for(; t + 1 < t_end; t += 2)
    {
        int phys0 = block_table[b * MAXB + t / BS];
        int off0  = t % BS;
        size_t row0 = ((size_t)(phys0 * BS + off0) * H + h) * D;
        const half* krow0 = k_cache + row0;
        const half* vrow0 = v_cache + row0;

        int t1    = t + 1;
        int phys1 = block_table[b * MAXB + t1 / BS];
        int off1  = t1 % BS;
        size_t row1 = ((size_t)(phys1 * BS + off1) * H + h) * D;
        const half* krow1 = k_cache + row1;
        const half* vrow1 = v_cache + row1;

        float s0 = 0.f, s1 = 0.f;
        for (int k = 0, d = lane; k < D / 32; ++k, d += 32)
        {
            float qv = __half2float(qrow[d]);
            s0 += qv * __half2float(krow0[d]);
            s1 += qv * __half2float(krow1[d]);
        }

        for (int offset = 16; offset > 0; offset >>= 1)
        {
            s0 += __shfl_down_sync(0xffffffff, s0, offset);
            s1 += __shfl_down_sync(0xffffffff, s1, offset);
        }
        s0 = __shfl_sync(0xffffffff, s0, 0);
        s1 = __shfl_sync(0xffffffff, s1, 0);

        float m_new = fmaxf(m, fmaxf(s0, s1));
        float alpha = __expf(m - m_new);
        float p0 = __expf(s0 - m_new);
        float p1 = __expf(s1 - m_new);

        denom = denom * alpha + p0 + p1;

        for (int k = 0, d = lane; k < D / 32; ++k, d += 32)
        {
            acc[k] = acc[k] * alpha + p0 * __half2float(vrow0[d]) + p1 * __half2float(vrow1[d]);
        }
        m = m_new;
    }

    if (t < t_end)
    {
        int phy = block_table[b * MAXB + t / BS];
        int off = t % BS;
        size_t row = ((size_t)(phy * BS + off) * H + h) * D;
        const half* k_row = k_cache + row;
        const half* v_row = v_cache + row;

        float s = 0.f;
        for (int k = 0, d = lane; k < D / 32; ++k, d += 32)
        {
            s += __half2float(qrow[d]) * __half2float(k_row[d]);
        }
        for (int offset = 16; offset > 0; offset >>= 1)
        {
            s += __shfl_down_sync(0xffffffff, s, offset);
        }
        s = __shfl_sync(0xffffffff, s, 0);

        float m_new = fmaxf(m, s);
        float alpha = __expf(m - m_new);
        float p = __expf(s - m_new);
        denom = denom * alpha + p;
        for (int k = 0, d = lane; k < D / 32; ++k, d += 32)
        {
            acc[k] = acc[k] * alpha + p * __half2float(v_row[d]);
        }
        m = m_new;
    }

    // ===== TODO 4：把局部结果写进 partial buffer（不要在这里除以 denom） =====
    // partial_m和partial_d 32个lane都是一个值，所以让lane 0 写就行了
    // 因为s广播到所有lane后，下面的m_new、alpha...都是用的同样的s
    if(lane == 0)
    {
        // partial_m和partial_d 形状是[NSPLIT, B, H]
        partial_m[split * B * H + b * H + h] = m;
        partial_d[split * B * H + b * H + h] = denom;
    }

    for(int k = 0, d = lane; k < D/32; ++k, d += 32)
    {
        // 每个 lane 写自己负责的 4 维
        // partial_acc 形状是 [NSPLIT, B, H, D]
        // 每个split是一个块完整的[B,H,D] ———— partial[i] 就是“第 i 个 split 的完整结果”
        // 这里完整的(B, H, D)是指每个 split 扫的是 L 的一段，所以它算出的 acc_i 是“部分结果”
        // 但这个部分结果覆盖的 (b,h,d) 范围是完整的，缺的只是 L 方向上的另外 3/4
        partial_acc[split * B * H * D + b * H * D + h * D + d] = acc[k];
    }

}

// ─────────────────────────────────────────────────────────────
// kernel 2：合并 NSPLIT 份部分结果
// ─────────────────────────────────────────────────────────────
__global__ void merge_kernel(const float* __restrict__ partial_acc,   // [NSPLIT,B,H,D]
                             const float* __restrict__ partial_m,     // [NSPLIT,B,H]
                             const float* __restrict__ partial_d,     // [NSPLIT,B,H]
                             half* __restrict__ out,                  // [B,H,D]
                             int B, int H, int D)
{
    // 不是按 split 分工，split 是在 warp 内部用 for (i) 循环消掉的
    int gwarp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;   // 一个 warp 一个 (b,h)
    int lane  = threadIdx.x & 31;

    if (gwarp >= B * H)
    {
        return;
    }

    int b = gwarp / H;
    int h = gwarp % H;

    // 读4个 split 的 m_i 和 d_i
    float m_arr[NSPLIT], d_arr[NSPLIT];
    for (int i = 0; i < NSPLIT; ++i)
    {
        m_arr[i] = partial_m[(i * B + b) * H + h];
        d_arr[i] = partial_d[(i * B + b) * H + h];
    }

    // 算全局最大M
    float M = -INFINITY;
    for (int i = 0; i < NSPLIT; ++i)
    {
         M = fmaxf(M, m_arr[i]);    // 4 个 split 里最大的那个分数
    }

    // ===== TODO 6：按 alpha = exp(m_i - M) 累加 d 和 acc =====
    float d = 0.f;
    float acc[4] = {0.f};
    for (int i = 0; i < NSPLIT; ++i) 
    {
        // 第 i 段的 d_i 和 acc_i 是按 m_i 归一化的
        // 全局基准是 M
        float alpha = __expf(m_arr[i] - M);     // 第 i 段的缩放因子
        d += d_arr[i] * alpha;                  // 分母加权累加
        for (int k = 0; k < D / 32; ++k)        // 分子逐维加权累加
        {
            acc[k] += partial_acc[((i * B + b) * H + h) * D + (lane + 32 * k)] * alpha;
        }
            
    }

    // ===== TODO 7：写回（和单链版的结尾一样，除以 d） =====
    float inv = (d > 0.f) ? 1.f / d : 0.f;
    for (int k = 0, dd = lane; k < D / 32; ++k, dd += 32)
    {
         out[b * H * D + h * D + dd] = __float2half(acc[k] * inv);
    }
       
}

// ─────────────────────────────────────────────────────────────
// host：分配 partial buffer → 启动两个 kernel
// ─────────────────────────────────────────────────────────────
torch::Tensor paged_attn_split(torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache,
                               torch::Tensor block_table, torch::Tensor context_lens)
{
    int B = q.size(0), H = q.size(1), D = q.size(2);
    int MAXB = block_table.size(1);

    auto out = torch::empty({B, H, D}, q.options());
    auto opts_f = q.options().dtype(torch::kFloat32);
    auto partial_acc = torch::empty({NSPLIT, B, H, D}, opts_f);         // [4, 8, 32, 128]      每段的未归一化分子
    auto partial_m   = torch::empty({NSPLIT, B, H}, opts_f);            // [4, 8, 32]           每段的最大分数
    auto partial_d   = torch::empty({NSPLIT, B, H}, opts_f);            // [4, 8, 32]           每段的 exp 和（分母）

    int threads = 256;

    int warps1 = B * H * NSPLIT;                                        // 8 * 32 * 4 = 1024 个warp
    int blocks1 = (warps1 * 32 + threads - 1) / threads;                // 128 个block

    paged_attn_split_kernel<<<blocks1, threads>>>(
        reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(k_cache.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(v_cache.data_ptr<at::Half>()),
        block_table.data_ptr<int>(),
        context_lens.data_ptr<int>(),
        partial_acc.data_ptr<float>(),
        partial_m.data_ptr<float>(),
        partial_d.data_ptr<float>(),
        B, H, D, MAXB);

    int warps2 = B * H;
    int blocks2 = (warps2 * 32 + threads - 1) / threads;

    merge_kernel<<<blocks2, threads>>>(
        partial_acc.data_ptr<float>(),
        partial_m.data_ptr<float>(),
        partial_d.data_ptr<float>(),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        B, H, D);

    return out;
}
