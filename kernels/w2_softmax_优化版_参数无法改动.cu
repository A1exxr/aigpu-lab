#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cfloat>

__global__ void softmax_kernel(const half* __restrict__ x,
                               half* __restrict__ out,
                               const int rows, const int cols){
    const int tid = threadIdx.x;

    __shared__ float x_exp[4096];     // 一行的 4096 个元素（16 KB）
    __shared__ float x_max[512];      // max 归约的中间结果
    __shared__ float sum_data[512];   // Σexp 归约的中间结果

    // ---------------------------------------------------------------
    // ① 搬进共享内存 —— 全 kernel 【唯一一次】全局读
    // ---------------------------------------------------------------
    for(int i = tid; i < cols; i += blockDim.x)
    {
        x_exp[i] = __half2float(x[blockIdx.x * cols + i]);
    }
    // ★ 这里【不需要】__syncthreads()，因为是单个线程的读写

    // ---------------------------------------------------------------
    // ② 求 max —— 只读 x_exp，不再碰全局内存
    // ---------------------------------------------------------------
    x_max[tid] = -FLT_MAX;
    for(int i = tid; i < cols; i += blockDim.x)
    {
        // fmax() 取两个浮点数的最大值
        x_max[tid] = fmaxf(x_max[tid], x_exp[i]);
    }
    __syncthreads();                  // 要有屏障，共享内存写完，先同步，再读取

    for(int i = blockDim.x / 2; i > 0; i >>= 1)
    {
        if(tid < i)
        {
            x_max[tid] = fmaxf(x_max[tid], x_max[tid + i]);
        }
        __syncthreads();
    }

    // ---------------------------------------------------------------
    // ③ 求 Σexp：原地把 x_exp 换成 exp 结果
    // ---------------------------------------------------------------
    float sum = 0.f;
    for(int i = tid; i < cols; i += blockDim.x)
    {
        float e = expf(x_exp[i] - x_max[0]);
        x_exp[i] = e;                 // 原始值在这里被覆盖，不再需要
        sum += e;
    }
    sum_data[tid] = sum;
    __syncthreads();

    for(int i = blockDim.x / 2; i > 0; i >>= 1)
    {
        if(tid < i)
        {
            sum_data[tid] += sum_data[tid + i];
        }
        __syncthreads();
    }

    // ---------------------------------------------------------------
    // ④ 归一化 —— 全 kernel 【唯一一次】全局写
    // ---------------------------------------------------------------
    for(int i = tid; i < cols; i += blockDim.x)
    {
        out[blockIdx.x * cols + i] = __float2half(x_exp[i] / sum_data[0]);
    }
}



torch::Tensor softmax_lastdim(torch::Tensor x){
    TORCH_CHECK(x.is_cuda(),"x 必须在GPU上");
    TORCH_CHECK(x.scalar_type() == torch::kFloat16);


    auto xc = x.contiguous();
    auto out = torch::empty_like(xc);
    
    const int cols = (int)xc.size(-1);
    const int rows = (int)(xc.numel() / cols);

    const int threads = 512;
    const int blocks = rows;
    softmax_kernel<<<blocks, threads>>>(
        reinterpret_cast<const half*>(xc.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        rows,cols
    );
    return out;
}