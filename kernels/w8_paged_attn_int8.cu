// w8 台阶1：PagedAttention decode + INT8 KV cache
// 基于 w7_paged_attn.cu（单链 2 路展开版），只换 K 的存储格式，算法不变。
//
// 核心：score(t,h) = scale(t,h) * Σ_d q[d] * k8[t,d]
//      每个token的每个头都有一个scale，一共有 2048 * 32 =65536个scale，  一个头有128维，共同用一个scale

#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cstdint>

#define BS   16     // 每块 token 数
#define DMAX 128    // 每个头的最大维度（DMAX/32 = 每线程最多 4 个元素）

__global__ void paged_attn_int8_kernel(const half*   __restrict__ q,            // [B,H,D]
                                       const int8_t* __restrict__ k_q,          // [NB,BS,H,D] 量化后的 K 是int8类型
                                       const half*   __restrict__ k_scale,      // [NB,BS,H]   每个 (token,head) 一个 scale
                                       const half*   __restrict__ v_cache,      // [NB,BS,H,D] V 未量化
                                       const int*    __restrict__ block_table,  // [B,MAXB]
                                       const int*    __restrict__ context_lens, // [B]
                                       half*         __restrict__ out,          // [B,H,D]
                                       int B, int H, int D, int MAXB)
{
    int gwarp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;   // 全局第几个 warp
    int lane  = threadIdx.x & 31;
    if (gwarp >= B * H) return;

    int b = gwarp / H;
    int h = gwarp % H;
    int L = context_lens[b];

    float acc[4] = {0.f};
    float denom  = 0.f;
    float m      = -INFINITY;

    const half* qrow = q + (size_t)(b * H * D + h * D);

    // t 在循环外声明：L 为奇数时尾块要用
    int t = 0;
    for (; t + 1 < L; t += 2)
    {
        // ---- token t ----
        int phys0 = block_table[b * MAXB + t / BS];
        int off0  = t % BS;
        size_t row0 = ((size_t)(phys0 * BS + off0) * H + h) * D;
        const int8_t* krow0 = k_q + row0;
        const half*   vrow0 = v_cache + row0; 
        size_t sidx0 = (size_t)(phys0 * BS + off0) * H + h;         // 计算该token的这个头的scale在k_scale的位置    [NB,BS,H]
        float  sc0   = __half2float(k_scale[sidx0]);

        // ---- token t+1 ----
        int t1    = t + 1;
        int phys1 = block_table[b * MAXB + t1 / BS];
        int off1  = t1 % BS;
        size_t row1 = ((size_t)(phys1 * BS + off1) * H + h) * D;
        const int8_t* krow1 = k_q + row1;
        const half*   vrow1 = v_cache + row1;
        size_t sidx1 = (size_t)(phys1 * BS + off1) * H + h;
        float  sc1   = __half2float(k_scale[sidx1]);

        // 两个点积交错：4 个独立 load 同时在飞
        float s0 = 0.f, s1 = 0.f;
        for (int k = 0, d = lane; k < D / 32; ++k, d += 32)
        {
            float qf = __half2float(qrow[d]);
            s0 += qf * (float)krow0[d];      // int8 → float，然后乘 fp16 转来的 q
            s1 += qf * (float)krow1[d];
        }

        // warp 归约 + 广播
        for (int offset = 16; offset > 0; offset >>= 1)
        {
            s0 += __shfl_down_sync(0xffffffff, s0, offset);
            s1 += __shfl_down_sync(0xffffffff, s1, offset);
        }
        s0 = __shfl_sync(0xffffffff, s0, 0);
        s1 = __shfl_sync(0xffffffff, s1, 0);

        // scale 在这里乘：整个 dot product 的公共因子
        s0 *= sc0;
        s1 *= sc1;

        // online softmax 一次吃两个分数
        float m_new = fmaxf(m, fmaxf(s0, s1));
        float alpha = __expf(m - m_new);
        float p0    = __expf(s0 - m_new);
        float p1    = __expf(s1 - m_new);

        denom = denom * alpha + p0 + p1;

        for (int k = 0, d = lane; k < D / 32; ++k, d += 32)
        {
            acc[k] = acc[k] * alpha + p0 * __half2float(vrow0[d])
                                   + p1 * __half2float(vrow1[d]);
        }
        m = m_new;
    }

    // L 为奇数：还剩最后一个 token
    if (t < L)
    {
        int phy = block_table[b * MAXB + t / BS];
        int off = t % BS;
        size_t row = ((size_t)(phy * BS + off) * H + h) * D;
        const int8_t* k_row = k_q + row;
        const half*   v_row = v_cache + row;
        size_t sidx = (size_t)(phy * BS + off) * H + h;
        float  sc   = __half2float(k_scale[sidx]);

        float s = 0.f;
        for (int k = 0, d = lane; k < D / 32; ++k, d += 32)
            s += __half2float(qrow[d]) * (float)k_row[d];

        for (int offset = 16; offset > 0; offset >>= 1)
            s += __shfl_down_sync(0xffffffff, s, offset);
        s = __shfl_sync(0xffffffff, s, 0);

        s *= sc;

        float m_new = fmaxf(m, s);
        float alpha = __expf(m - m_new);
        float p     = __expf(s - m_new);

        denom = denom * alpha + p;

        for (int k = 0, d = lane; k < D / 32; ++k, d += 32)
            acc[k] = acc[k] * alpha + p * __half2float(v_row[d]);
        m = m_new;
    }

    // 写回：acc / denom
    float inv = (denom > 0.f) ? 1.0f / denom : 0.0f;
    for (int k = 0, d = lane; k < D / 32; ++k, d += 32)
        out[b * H * D + h * D + d] = __float2half(acc[k] * inv);
}


torch::Tensor paged_attn_int8(torch::Tensor q, torch::Tensor k_q, torch::Tensor k_scale,
                             torch::Tensor v_cache, torch::Tensor block_table,
                             torch::Tensor context_lens)
{
    int B = q.size(0), H = q.size(1), D = q.size(2);
    int MAXB = block_table.size(1);
    auto out = torch::empty({B, H, D}, q.options());

    int warps   = B * H;                                 // 一个 warp 一对 (b,h)
    int threads = 256;                                   // 8 warp / block
    int blocks  = (warps * 32 + threads - 1) / threads;

    paged_attn_int8_kernel<<<blocks, threads>>>(
        reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
        k_q.data_ptr<int8_t>(),
        reinterpret_cast<const half*>(k_scale.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(v_cache.data_ptr<at::Half>()),
        block_table.data_ptr<int>(),
        context_lens.data_ptr<int>(),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        B, H, D, MAXB);
    return out;
}
