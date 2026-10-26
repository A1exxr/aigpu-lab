// w8 台阶2：PagedAttention decode + INT8 KV cache（K 和 V 都量化）
// 基于 w8_paged_attn_int8.cu（台阶1：只量化 K）。
//
// 与台阶1 的唯一区别：V 也存 int8，但它的 scale 折进权重 p，不进内层循环。
//   out += p * (sv * v8) = (p * sv) * v8 = pc * v8
// 每元素仍是 1 次乘加，字节数从 2 降到 1

// 这个是台阶3，在台阶2 KV量化基础上，再实现 ？ 的向量化访存

#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cstdint>

#define BS   16     // 每块 token 数
#define DMAX 128    // 每个头的最大维度（DMAX/32 = 每线程最多 4 个元素）

__global__ void paged_attn_int8kv_kernel(const half*   __restrict__ q,            // [B,H,D]
                                         const int8_t* __restrict__ k_q,          // [NB,BS,H,D] 量化后的 K
                                         const half*   __restrict__ k_scale,      // [NB,BS,H]   每个 (token,head) 一个 scale
                                         const int8_t* __restrict__ v_q,          // [NB,BS,H,D] 量化后的 V
                                         const half*   __restrict__ v_scale,      // [NB,BS,H]   V 的 scal 
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
        const int8_t* vrow0 = v_q + row0;    // K/V 同一套物理布局，row 完全相同
        size_t sidx0 = (size_t)(phys0 * BS + off0) * H + h;   // [NB,BS,H]，分母是 H 不是 D
        float  sc0   = __half2float(k_scale[sidx0]);
        float  sv0   = __half2float(v_scale[sidx0]);          // 同一个下标，不同数组

        // ---- token t+1 ----
        int t1    = t + 1;
        int phys1 = block_table[b * MAXB + t1 / BS];
        int off1  = t1 % BS;
        size_t row1 = ((size_t)(phys1 * BS + off1) * H + h) * D;
        const int8_t* krow1 = k_q + row1;
        const int8_t* vrow1 = v_q + row1;
        size_t sidx1 = (size_t)(phys1 * BS + off1) * H + h;
        float  sc1   = __half2float(k_scale[sidx1]);
        float  sv1   = __half2float(v_scale[sidx1]);

        // 这里实现向量化访存
        float s0 = 0.f, s1 = 0.f;
        for(int kb = 0; kb < D/128; ++kb)           // D = 128 时只跑一次
        {
            int off4 = kb * 128 + lane * 4;

            int k4a = *reinterpret_cast<const int*>(krow0 + off4);  // token t
            int k4b = *reinterpret_cast<const int*>(krow1 + off4);  // token t+1

            const __half2* q2 = reinterpret_cast<const __half2*>(qrow + off4);
            float2 qa = __half22float2(q2[0]);
            float2 qb = __half22float2(q2[1]);  

                s0 += qa.x*(int8_t)(k4a)       + qa.y*(int8_t)(k4a >> 8)
                    + qb.x*(int8_t)(k4a >> 16) + qb.y*(int8_t)(k4a >> 24);
                s1 += qa.x*(int8_t)(k4b)       + qa.y*(int8_t)(k4b >> 8)
                    + qb.x*(int8_t)(k4b >> 16) + qb.y*(int8_t)(k4b >> 24);
        }

        // warp 归约 + 广播
        for (int offset = 16; offset > 0; offset >>= 1)
        {
            s0 += __shfl_down_sync(0xffffffff, s0, offset);
            s1 += __shfl_down_sync(0xffffffff, s1, offset);
        }
        s0 = __shfl_sync(0xffffffff, s0, 0);
        s1 = __shfl_sync(0xffffffff, s1, 0);

        // K 的 scale 在这里乘：整个 dot product 的公共因子
        s0 *= sc0;
        s1 *= sc1;

        // online softmax 一次吃两个分数
        float m_new = fmaxf(m, fmaxf(s0, s1));
        float alpha = __expf(m - m_new);
        float p0    = __expf(s0 - m_new);
        float p1    = __expf(s1 - m_new);

        denom = denom * alpha + p0 + p1;

        // V 的 scale 折进 p：pc = p * sv。每 token 一次，内层不变
        float pc0 = p0 * sv0;
        float pc1 = p1 * sv1;

        for(int kb = 0; kb < D/128; ++kb)
        {
            int off4 = kb * 128 + lane * 4;
            
            int v4a = *reinterpret_cast<const int*>(vrow0 + off4);
            int v4b = *reinterpret_cast<const int*>(vrow1 + off4);

            acc[0] = acc[0]*alpha + pc0*(int8_t)(v4a)        + pc1*(int8_t)(v4b);
            acc[1] = acc[1]*alpha + pc0*(int8_t)(v4a >> 8)   + pc1*(int8_t)(v4b >> 8);
            acc[2] = acc[2]*alpha + pc0*(int8_t)(v4a >> 16)  + pc1*(int8_t)(v4b >> 16);
            acc[3] = acc[3]*alpha + pc0*(int8_t)(v4a >> 24)  + pc1*(int8_t)(v4b >> 24);
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
        const int8_t* v_row = v_q + row;
        size_t sidx = (size_t)(phy * BS + off) * H + h;
        float  sc   = __half2float(k_scale[sidx]);
        float  sv   = __half2float(v_scale[sidx]);

        float s = 0.f;
        for (int kb = 0; kb < D / 128; ++kb)
        {
            int off4 = kb * 128 + lane * 4;
            int k4 = *reinterpret_cast<const int*>(k_row + off4);
            const __half2* q2 = reinterpret_cast<const __half2*>(qrow + off4);
            float2 qa = __half22float2(q2[0]);
            float2 qb = __half22float2(q2[1]);
            s += qa.x*(int8_t)(k4)       + qa.y*(int8_t)(k4 >> 8)
                + qb.x*(int8_t)(k4 >> 16) + qb.y*(int8_t)(k4 >> 24);
        }


        for (int offset = 16; offset > 0; offset >>= 1)
        {
            s += __shfl_down_sync(0xffffffff, s, offset);
        }
            
        s = __shfl_sync(0xffffffff, s, 0);

        s *= sc;

        float m_new = fmaxf(m, s);
        float alpha = __expf(m - m_new);
        float p     = __expf(s - m_new);
        float pc    = p * sv;      // 尾块同样要折 scale

        denom = denom * alpha + p;

        for (int kb = 0; kb < D / 128; ++kb)
        {
            int off4 = kb * 128 + lane * 4;
            int v4 = *reinterpret_cast<const int*>(v_row + off4);
            acc[0] = acc[0]*alpha + pc*(int8_t)(v4);
            acc[1] = acc[1]*alpha + pc*(int8_t)(v4 >> 8);
            acc[2] = acc[2]*alpha + pc*(int8_t)(v4 >> 16);
            acc[3] = acc[3]*alpha + pc*(int8_t)(v4 >> 24);
        }
        m = m_new;
    }

    // 写回：acc / denom
    float inv = (denom > 0.f) ? 1.0f / denom : 0.0f;
    for (int kb = 0; kb < D / 128; ++kb)
    {
        for (int i = 0; i < 4; ++i)
        {
            out[b*H*D + h*D + kb*128 + lane*4 + i] = __float2half(acc[i] * inv);
        }
    }
}


torch::Tensor paged_attn_int8kv(torch::Tensor q, torch::Tensor k_q, torch::Tensor k_scale,
                                torch::Tensor v_q, torch::Tensor v_scale,
                                torch::Tensor block_table, torch::Tensor context_lens)
{
    int B = q.size(0), H = q.size(1), D = q.size(2);
    int MAXB = block_table.size(1);
    auto out = torch::empty({B, H, D}, q.options());

    int warps   = B * H;                                 // 一个 warp 一对 (b,h)
    int threads = 256;                                   // 8 warp / block
    int blocks  = (warps * 32 + threads - 1) / threads;

    paged_attn_int8kv_kernel<<<blocks, threads>>>(
        reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
        k_q.data_ptr<int8_t>(),
        reinterpret_cast<const half*>(k_scale.data_ptr<at::Half>()),
        v_q.data_ptr<int8_t>(),
        reinterpret_cast<const half*>(v_scale.data_ptr<at::Half>()),
        block_table.data_ptr<int>(),
        context_lens.data_ptr<int>(),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        B, H, D, MAXB);
    return out;
}
