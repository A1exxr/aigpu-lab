/* 骨架：LayerNorm 反向 · 台阶1（只算 dx）
   给定 x, dy, gamma, eps，求 dx。dγ/dβ 留到台阶2。

   正向： μ = mean(x)    σ = sqrt(var(x)+eps)      x̂ = (x−μ)/σ       y = γ·x̂ + β
   反向： dx̂_j = γ_j·dy_j                      （先把梯度退回 x̂）
         dx_j  = (1/σ)·[ dx̂_j − mean(dx̂) − x̂_j·mean(dx̂·x̂) ]
                      └────┬────┘        └──────┬──────┘
                     行内标量 s1          行内标量 s2
    两个 mean 都在 dx̂ 上求，也就是【带 γ】的：
        s1 = mean(γ·dy)        s2 = mean(γ·dy·x̂)
     把 γ 提到 mean 外面是错的（只有 γ 全相等时才碰巧对）。

   注意：要算 x̂ 得先有 μ 和 σ → 还需要两个行内归约（Σx、Σx²）。
   所以这个 kernel 一共 4 次行内归约：
       ① Σx     ② Σx²     ③ Σ(γ·dy)      ④ Σ(γ·dy·x̂)
   访存：读 x(N) + 读 dy(N) + 写 dx(N) = 3N（gamma 只有 D 个元素，可忽略）
        理论下限 = 3N / 221.4 GB/s，(2048,4096) fp16 → 48 MiB → 0.227 ms

*/

#include <cuda_fp16.h>
#include <torch/extension.h>

#define SM_THREADS 256
#define ELEMS_PER_THREAD 16          // D = 4096 时 256*16 = 4096
#define WARP_NUM (SM_THREADS / 32)


// 这个反向传播会归约很多，把两级归约写成一个device函数
// device必须写到kernel外面
// __forceincline__ 是强制内联；告诉编译器“把这个函数直接展开到调用处，不要生成函数调用”
// 定义要放在调用它的kernel之前
__device__ __forceinline__ float block_reduce(float v, float* smem)
{
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    // 开头加一个屏障，保证上一次block_reduce的结果smem[0]已经被所有线程读完
    __syncthreads();

    for(int offset = 16; offset > 0; offset >>= 1)
    {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    
    if(lane == 0)
    {
        smem[warp_id] = v;
    }
    __syncthreads();

    if(warp_id == 0)
    {
        float k = (lane < WARP_NUM) ? smem[lane] : 0.f;
        for(int offset = WARP_NUM / 2; offset > 0; offset >>= 1)
        {   
            k += __shfl_down_sync(0xffffffff, k, offset);
        }
        if(lane == 0)
        {
            smem[0] = k;
        }
    }
    __syncthreads();

    return smem[0];
}


__global__ void layernorm_backward_kernel(
                                        const half* __restrict__ dy,     // [rows, D]
                                        const half* __restrict__ x,      // [rows, D]
                                        const half* __restrict__ gamma,  // [D]
                                        half* __restrict__ dx,           // [rows, D]
                                        int rows, int D, float eps)
{
    // 一个block处理一行4096个元素，共256个线程，每个线程处理16个元素， 8个warp
    // 定位 + 把 x/dy 读进寄存器（16 + 16 个 half = 16 个寄存器）
    int row = blockIdx.x;
    size_t base = (size_t)row * D;
    int tid = threadIdx.x;

    // 读x、dy到寄存器，同时累加x和x方
    float xreg[16] = {0.f};
    float dyreg[16] = {0.f};
    float x_sum = 0.f;
    float x2_sum = 0.f;

    #pragma unroll
    for(int i = 0; i < 16; ++i)
    {
        int j = i * 256 + tid;
        xreg[i] = __half2float(x[base + j]);
        dyreg[i] = __half2float(dy[base + j]);
        x_sum += xreg[i];               // 局部和
        x2_sum += xreg[i] * xreg[i];    // 局部平方和
    }

    __shared__ float smem[WARP_NUM];
    // 两次归约
    float sum_x = block_reduce(x_sum, smem);
    float sum_x2 = block_reduce(x2_sum, smem);


    // 算 μ、σ、x̂
    float mu = sum_x / D;
    float var = sum_x2 / D - mu * mu;
    float inv_sigma = rsqrtf(var + eps);

    // 算 x̂， dx̂ = dy*γ,记作 gdy ，同时累加 gdy 和 gdy*x̂  两个行内标量
    float xhat[16];
    float gdy[16];
    float s1_part = 0.f;
    float s2_part = 0.f;

    #pragma unroll
    for(int i = 0; i < ELEMS_PER_THREAD; ++i)
    {
        int j = tid + i * SM_THREADS;
        xhat[i] = (xreg[i] - mu) * inv_sigma;
        float g = __half2float(gamma[j]);
        gdy[i] = g * dyreg[i];
        s1_part += gdy[i];
        s2_part += gdy[i] * xhat[i];
    }

    // 两次归约得到 gdy 和 gdy*x̂ 总和，再除以D，得到mean
    float s1 = block_reduce(s1_part, smem) / D;
    float s2 = block_reduce(s2_part, smem) / D;

    // 写回
    #pragma unroll
    for(int i = 0; i < ELEMS_PER_THREAD; ++i)
    {
        int j = tid + i * SM_THREADS;
        dx[base + j] = __float2half(inv_sigma * (gdy[i] - s1 - xhat[i] * s2));
    }
}
  



torch::Tensor layernorm_backward(
    torch::Tensor dy, torch::Tensor x, torch::Tensor gamma, double eps)
{
    auto rows = x.size(0);
    auto D = x.size(-1);
    auto dx = torch::empty_like(x);

    layernorm_backward_kernel<<<rows, SM_THREADS>>>(
        reinterpret_cast<const half*>(dy.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(gamma.data_ptr<at::Half>()),
        reinterpret_cast<half*>(dx.data_ptr<at::Half>()),
        (int)rows, (int)D, (float)eps);

    return dx;
}
