// w8_paged_attn_int8kvsplit.cu
// W8 台阶4：KV 全 int8（单字节 load）+ split-L
//
// 为什么先做这一格：
//   台阶2 = KV int8 + 单字节 + grid 32 → 1.2032 ms / 113.4 GB/s（51%）
//   台阶3 = KV int8 + vec4   + grid 32 → 1.5503 ms /  88.0 GB/s（41%）
//   两者都比台阶1（K-only，91%）差，假设是「grid 只有 32 个 block，没 warp 盖延迟」。
//   台阶4 **只动并行度**（把 vec4 撤回去，单字节保留），grid 32 → 128：
//     若回到 ~85%  ⇒ 并行度假设成立，台阶2 的慢是延迟问题
//     若还是 ~50%  ⇒ 单字节指令数太多是主因（每 token 8 条 vs 2 条）
//   这样台阶5（vec4 + split-L）才有干净的对照。
//
// 2×2 对照表（这是整个 w8 的实验设计）：
//                 单字节               vec4
//   grid 32   台阶2  1.2032 / 51%    台阶3  1.5503 / 41%
//   grid 128  台阶4  ←本次           台阶5  （下一步）
//
// 配置：B=8, H=32, D=128, L=2048, BS=16, MAXB=128, NSPLIT=4
// 流量：130.1 MiB + partial 往返 2 MiB；理论下限 130.1 MiB / 213 GB/s ≈ 0.64 ms

#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#define BS        16
#define NSPLIT     4
#define SM_THREADS 256

__global__ void paged_attn_int8kvsplit_kernel(
                                            const half* __restrict__ q,            // [B,H,D]
                                            const int8_t* __restrict__ k_q,        // [NB,BS,H,D]
                                            const half* __restrict__ k_scale,      // [NB,BS,H]
                                            const int8_t* __restrict__ v_q,        // [NB,BS,H,D]
                                            const half* __restrict__ v_scale,      // [NB,BS,H]
                                            const int* __restrict__ block_table,   // [B,MAXB]
                                            const int* __restrict__ context_lens,  // [B]
                                            float* __restrict__ partial_acc,       // [NSPLIT,B,H,D]
                                            float* __restrict__ partial_m,         // [NSPLIT,B,H]
                                            float* __restrict__ partial_denom,     // [NSPLIT,B,H]
                                            int B, int H, int D, int MAXB)
{
    int gwarp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane  = threadIdx.x & 31;
    int nbh   = B * H;

    int bh = gwarp / NSPLIT;                // 0 1 2 ... 255
    int split = gwarp % NSPLIT;                            // 0 1 2 3
    if(bh >= B * H)
    {
        return;
    }

    int b = bh / H;
    int h = bh % H;
    int L = context_lens[b];
    int chunk = (L + NSPLIT - 1) / NSPLIT;
    int t_begin = split * chunk;
    int t_end = min(L, t_begin + chunk);

    float m = -INFINITY;
    float denom = 0.f;
    float acc[4] = {0.f};

    const half* qrow = q + size_t(b * H * D + h * D);
    int t = t_begin;
    for(; t + 1 < t_end; t += 2)
    {
        int phys0 = block_table[b * MAXB + t / BS];
        int off0 = t % BS;
        size_t row0 = (size_t)(phys0 * BS * H * D + off0 * H * D + h * D);
        const int8_t* krow0 = k_q + row0;
        const int8_t* vrow0 = v_q + row0;
        size_t sidx0 = (size_t)(phys0 * BS * H + off0 * H + h);
        float k_s0 = __half2float(k_scale[sidx0]);
        float v_s0 = __half2float(v_scale[sidx0]);

        int t1 = t + 1;
        int phys1 = block_table[b * MAXB + t1 / BS];
        int off1 = t1 % BS;
        size_t row1 = (size_t)(phys1 * BS * H * D + off1 * H * D + h * D);
        const int8_t* krow1 = k_q + row1;
        const int8_t* vrow1 = v_q + row1;
        size_t sdix1 = (size_t)(phys1 * BS * H + off1 * H + h);
        float k_s1 = __half2float(k_scale[sdix1]);
        float v_s1 = __half2float(v_scale[sdix1]);

        float s0 = 0.f, s1 = 0.f;
        for(int k = 0, d = lane; k < D/32; ++k, d += 32)
        {
            float qv = __half2float(qrow[d]);
            s0 += qv * (float)(krow0[d]);           // INT8 数据应该直接转 float：
            s1 += qv * (float)(krow1[d]);
        }
        

        for(int offset = 16; offset > 0; offset >>= 1)
        {
            s0 += __shfl_down_sync(0xffffffff, s0, offset);
            s1 += __shfl_down_sync(0xffffffff, s1, offset);
        }
        s0 = __shfl_sync(0xffffffff, s0, 0);
        s1 = __shfl_sync(0xffffffff, s1, 0);
        s0 *= k_s0;
        s1 *= k_s1;

        float m_new = fmaxf(m, fmaxf(s0, s1));
        float alpha = __expf(m - m_new);
        float p0 = __expf(s0 - m_new);
        float p1 = __expf(s1 - m_new);
        denom = denom * alpha + p0 + p1;
        p0 *= v_s0;
        p1 *= v_s1;
        for(int k = 0, d = lane; k < D/32; ++k, d += 32)
        {
            acc[k] = acc[k] * alpha + p0 * (float)(vrow0[d]) + p1 * (float)(vrow1[d]);
        }
        m = m_new;
    }
    
    if(t < t_end)
    {
        int phys = block_table[b * MAXB + t / BS];
        int off = t % BS;
        size_t row = (size_t)(phys * BS * H * D + off * H * D + h * D);
        const int8_t* krow = k_q + row;
        const int8_t* vrow = v_q + row;
        size_t sidx = (size_t)(phys * BS * H + off * H + h);
        float k_s = __half2float(k_scale[sidx]);
        float v_s = __half2float(v_scale[sidx]);

        float s = 0;
        for(int k = 0, d = lane; k < D/32; ++k, d += 32)
        {
            s += __half2float(qrow[d]) * (float)(krow[d]);
        }
        
        for(int offset = 16; offset > 0; offset >>= 1)
        {
            s += __shfl_down_sync(0xffffffff, s, offset);
        }
        s = __shfl_sync(0xffffffff, s, 0);
        s *= k_s;

        float m_new = fmaxf(m, s);
        float alpha = __expf(m - m_new);
        float p = __expf(s - m_new);
        denom = denom * alpha + p;
        p *= v_s;
        for(int k = 0, d = lane; k < D/32; ++k, d += 32)
        {
            acc[k] = acc[k] * alpha + p * (float)(vrow[d]);
        }
        m = m_new;
    }   

    if(lane == 0)
    {
        partial_m[split * B * H + b * H + h] = m;
        partial_denom[split * B * H + b * H + h] = denom;
    }
    
    for(int k = 0, d = lane; k < D/32; ++k, d += 32)
    {
        partial_acc[split * B * H * D + b * H * D + h * D + d] = acc[k];
    }
}

__global__ void merge_kernel(
                            const float* __restrict__ partial_acc,
                            const float* __restrict__ partial_m,
                            const float* __restrict__ partial_denom,
                            half* __restrict__ out,
                            int B, int H, int D)
{
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    if(warp >= B * H)
    {
        return;
    }

    int b = warp / H;
    int h = warp % H;
    float m_arr[NSPLIT] = {0.f};
    float d_arr[NSPLIT] = {0.f};
    for(int i = 0; i < NSPLIT; ++i)
    {
        m_arr[i] = partial_m[i * B * H + b * H + h];
        d_arr[i] = partial_denom[i * B * H + b * H + h];
    }

    float M = -INFINITY;
    for(int i = 0; i < NSPLIT; ++i)
    {
        M = fmaxf(M, m_arr[i]);
    }

    float denom = 0.f;
    float acc[4] = {0.f};
    for(int i = 0; i < NSPLIT; ++i)
    {
        float alpha = __expf(m_arr[i] - M);
        denom += d_arr[i] * alpha; 
        for(int k = 0; k < D/32; ++k)
        {
            acc[k] += partial_acc[i * B * H * D + b * H * D + h * D + lane + 32 * k] * alpha;
        }
    }

    float inv = (denom > 0.f) ? 1.0/denom : 0.f;
    for(int k = 0, d = lane; k < D/32; ++k, d += 32)
    {
        out[b * H * D + h * D + d] = __float2half(acc[k] * inv);
    }
}

torch::Tensor paged_attn_int8kvsplit(torch::Tensor q, torch::Tensor k_q,
                                     torch::Tensor k_scale, torch::Tensor v_q,
                                     torch::Tensor v_scale, torch::Tensor block_table,
                                     torch::Tensor context_lens)
{
    int B = q.size(0), H = q.size(1), D = q.size(2);
    int MAXB = block_table.size(1);
    auto opts_f = q.options().dtype(torch::kFloat32);

    auto partial_acc   = torch::empty({NSPLIT, B, H, D}, opts_f);
    auto partial_m     = torch::empty({NSPLIT, B, H}, opts_f);
    auto partial_denom = torch::empty({NSPLIT, B, H}, opts_f);
    auto out = torch::empty({B, H, D}, q.options());

    int warps  = B * H * NSPLIT;                                // 8 * 32 * 4 = 1024个warp
    int blocks = (warps * 32 + SM_THREADS - 1) / SM_THREADS;    // 128个block

    paged_attn_int8kvsplit_kernel<<<blocks, SM_THREADS>>>(
        reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
        k_q.data_ptr<int8_t>(),
        reinterpret_cast<const half*>(k_scale.data_ptr<at::Half>()),
        v_q.data_ptr<int8_t>(),
        reinterpret_cast<const half*>(v_scale.data_ptr<at::Half>()),
        block_table.data_ptr<int>(),
        context_lens.data_ptr<int>(),
        partial_acc.data_ptr<float>(),
        partial_m.data_ptr<float>(),
        partial_denom.data_ptr<float>(),
        B, H, D, MAXB);

    int merge_blocks = (B * H * 32 + 127) / 128;        // 64
    merge_kernel<<<merge_blocks, 128>>>(
        partial_acc.data_ptr<float>(),
        partial_m.data_ptr<float>(),
        partial_denom.data_ptr<float>(),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        B, H, D);

    cudaDeviceSynchronize();
    return out;
}
