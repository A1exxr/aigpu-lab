#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cfloat>
#include <cstdlib>      // ← 新增（getenv / atoi）

    // 这个版本是可以改变参数的
    // 命令
    // cd ~/projects/aigpu-lab
    // for t in 128 256 512 1024; do
    // printf 'threads=%-6s' "$t"
    // SM_THREADS=$t python kernels/bench_kernels.py --only softmax 2>&1 | grep -E '\[(PASS|WRONG|FAIL)\]'
    // done


__global__ void softmax_kernel(const half* __restrict__ x,
                               half* __restrict__ out,
                               const int rows, const int cols){
    const int tid = threadIdx.x;

    // __shared__ float x_exp[4096];     // 一行的 4096 个元素（16 KB）
    // __shared__ float x_max[512];      // max 归约的中间结果
    // __shared__ float sum_data[512];   // Σexp 归约的中间结果

    extern __shared__ float smem[];                    // 动态共享内存（大小在启动时传），一个kernel只能申请一块，想用多块动态共享内存，只能通过指针偏移来划分
    float* x_exp    = smem;                            // [cols]
    float* x_max    = smem + cols;                     // [blockDim.x]
    float* sum_data = smem + cols + blockDim.x;        // [blockDim.x]

    // ---------------------------------------------------------------
    // ① 搬进共享内存 —— 全 kernel 【唯一一次】全局读
    // ---------------------------------------------------------------
    for(int i = tid; i < cols; i += blockDim.x)
    {
        x_exp[i] = __half2float(x[blockIdx.x * cols + i]);
    }
    // ★ 这里【不需要】__syncthreads()，原因见问题 1

    // ---------------------------------------------------------------
    // ② 求 max —— 只读 x_exp，不再碰全局内存
    // ---------------------------------------------------------------
    x_max[tid] = -FLT_MAX;
    for(int i = tid; i < cols; i += blockDim.x)
    {
        x_max[tid] = fmaxf(x_max[tid], x_exp[i]);
    }
    __syncthreads();                  // ★ 必须有，原因见问题 3

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

    const char* env = std::getenv("SM_THREADS");
    const int threads = env ? std::atoi(env) : 512;
    const int blocks = rows;

    const size_t smem_bytes = (size_t)(cols + 2 * threads) * sizeof(float);

    softmax_kernel<<<blocks, threads, smem_bytes>>>(
        reinterpret_cast<const half*>(xc.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        rows,cols
    );
    return out;
}