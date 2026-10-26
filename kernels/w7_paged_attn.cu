// 这是2路展开的版本：同时处理两个 token，维护两套互相独立的状态，让硬件在等 token0 的归约时去算 token1

#include <torch/extension.h>
#include <cuda_fp16.h>

#define BS   16     // 每块 token 数
#define DMAX 128    // 每线程最多负责 DMAX/32=4 个元素        每个头的最大维度是128

 
__global__ void paged_attn_kernel(const half* __restrict__ q,             // [B,H,D]
                                    const half* __restrict__ k_cache,       // [NB,BS,H,D]
                                    const half* __restrict__ v_cache,       // [NB,BS,H,D]
                                    const int*  __restrict__ block_table,   // [B,MAXB] 行主序：block_table[b*MAXB + 逻辑块号]
                                    const int*  __restrict__ context_lens,  // [B]
                                    half*       __restrict__ out,           // [B,H,D]
                                    int B, int H, int D, int MAXB)          // MAXB表示 每个请求最多能有多少个逻辑块
{
    int gwarp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;       // 全局第几个warp， 0-255
    int lane = threadIdx.x & 31;                                    // lane
    if(gwarp >= B * H)                                              // 跨界退出
    {
        return;
    }

    int b = gwarp / H;                  // 第几个请求
    int h = gwarp % H;                  // 第几个头
    int L = context_lens[b];            // 这个请求的真实长度

    float acc[4] = {0.f};               // 分子，4个维度的部分和
    float denom = 0.f;                  // 分母
    float m = -INFINITY;                // 初始化最大值

    // 当前(b,h)的query指针
    const half* qrow = q + (size_t)(b * H * D + h * D);

    // 两个token并行计算，需要两套地址计算
    // 因为 t 和 t+1 可能在不同的物理块里（只有 t % 16 != 15 时才同块）
    int t = 0;  // 若L是单数，还剩下个token要算，t就在for循环外面声明
    for(; t + 1 < L; t += 2)       
    {   
        // token t
        int phys0 = block_table[b * MAXB + t / BS];     
        int off0 = t % BS;                               
        size_t row0 = ((size_t)(phys0 * BS + off0) * H + h) * D;
        const half* krow0 = k_cache + row0;
        const half* vrow0 = v_cache + row0;

        // token t1
        int t1 = t + 1;
        int phys1 = block_table[b * MAXB + t1 / BS];
        int off1 = t1 % BS;
        size_t row1 = ((size_t)(phys1 * BS + off1) * H + h) * D;
        const half* krow1 = k_cache + row1;
        const half* vrow1 = v_cache + row1;

        // 两个点积交错进行
        float s0 = 0.f, s1 = 0.f;     
        for(int k = 0, d = lane; k < D/32; ++k, d += 32)      
        {
            float q = __half2float(qrow[d]);
            s0 += q * __half2float(krow0[d]);
            s1 += q * __half2float(krow1[d]);
        }

        // 两个 warp 归约
        for(int offset = 16; offset > 0; offset >>= 1)
        {
            s0 += __shfl_down_sync(0xffffffff, s0, offset);
            s1 += __shfl_down_sync(0xffffffff, s1, offset);
        }

        s0 = __shfl_sync(0xffffffff, s0, 0);      
        s1 = __shfl_sync(0xffffffff, s1, 0);      

        // online softmax 一次要吃两个分数
        // 不是分别做两次更新，而是一次性把两个分数都纳入
        float m_new = fmaxf(m, fmaxf(s0, s1));
        float alpha = __expf(m - m_new);        // 还是缩放一次，不用两次
        // 两个token各自算自己的p
        float p0 = __expf(s0 - m_new);
        float p1 = __expf(s1 - m_new);

        // 分母一次加两项
        denom = denom * alpha + p0 + p1;

        // acc也一次性加两项
        for(int k = 0, d = lane; k < D/32; ++k, d += 32)
        {   
            acc[k] = acc[k] * alpha + p0 * __half2float(vrow0[d]) + p1 * __half2float(vrow1[d]);        
        }
        m = m_new;
    }

    // 若L是奇数，还剩一个token没算
    if (t < L)
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

    // L轮结束后，这个query与历史K_cache就算完了，需要写回attention的输出向量
    float inv = (denom > 0.f) ? 1.0f / denom : 0.0f;

    for(int k = 0, d = lane; k < D/32; ++k, d +=32)
    {
        out[b * H * D + h * D + d] = __float2half(acc[k] * inv);
    }
}



torch::Tensor paged_attn(torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache,
                         torch::Tensor block_table, torch::Tensor context_lens)
{
    int B = q.size(0), H = q.size(1), D = q.size(2);
    int MAXB = block_table.size(1);
    auto out = torch::empty({B, H, D}, q.options());

    int warps = B * H;                              // 一个 warp 一对 (b,h)
    int threads = 256;                              // 8 warp / block
    int blocks = (warps * 32 + threads - 1) / threads;

    paged_attn_kernel<<<blocks, threads>>>(
        reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(k_cache.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(v_cache.data_ptr<at::Half>()),
        block_table.data_ptr<int>(),
        context_lens.data_ptr<int>(),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        B, H, D, MAXB);
    cudaDeviceSynchronize();
    return out;
}
