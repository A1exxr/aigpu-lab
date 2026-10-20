/**
 *  融合 RMSNorm + 残差
 *  shape (1, 2048, 4096)、fp16
 *  kernel1: h = x + res        （读 x, 读 res, 写 h）
 *  kernel2: out = rmsnorm(h)   （读 h, 写 out）
 *  为什么可以融合：
 *      数据一一对应：kernel2 的每个输入元素就是 kernel1 输出的同一个元素，位置不变，不需要重排。     
 *      中间结果够小：一个 block 负责一行 4096 个元素，256 个线程，每个线程只经手 16 个 → 16 个数塞寄存器毫无压力。
 *      如果每线程要经手 4096 个，寄存器装不下，融合就失败（这就是为什么大的中间结果必须落显存）。
 *      归约范围是"行内"：RMSNorm 的均值只在这一行里算，一个 block = 一行，归约在 block 内闭合，不需要跨 block 通信。如果需要跨 block 归约（比如整矩阵求和），一个 kernel 内做不完。
 * 
 *  重要：中间结果能驻留在片上的，就别让它回显存
 * 
 * 
*/

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>


// 这是融合版
__global__ void fused_rmsnorm_residual_kernel(const half* __restrict__ x,
                                            const half* __restrict__ res,
                                            const half* __restrict__ w,
                                            half* __restrict__ out,
                                            const int rows, const int cols, float eps){
    // 一个线程处理16个元素，要读x，res，相加后存入共享内存h里，然后再平方加和，归约算rstd
    // 2048 * 4096 = 2048 * 256 * 16
    const int tid = threadIdx.x;
    const int lane = threadIdx.x % 32;              // 0-15
    const int warp_id = threadIdx.x / 32;           // 0-7
    // 存放x和res的和
    // 不用申请共享内存来拷贝x的值，因为中间值都可以放到寄存器里
    float h[16];       // h[16] 是每个线程私有的局部数组
    float sum = 0.f;      
    // __shared__ float warp_sum[blockDim.x/32];        // blockDim.x是运行时才能确定的值，这里必须放编译期常量
    __shared__ float warp_sum[256/32];
    float temp;         // 需要一个标量来存值，不能用共享内存进行warp内部的归约
    __shared__ float rstd;
    #pragma unroll      // 加这个，就能把下面的循环打开，就可以分别把值存入
    for(int i = 0; i < 16; ++i)
    {
        int offset = blockIdx.x * cols + blockDim.x * i + tid;
        temp = __half2float(x[offset]) + __half2float(res[offset]);
        h[i] = temp;         // 现在h[0-15]是 该线程 要处理的16个元素(x + res)
        sum += temp * temp;   // sum是线程要处理的16个元素的 平方和
    }
    // 不需要屏障，不涉及共享内存

    // 归约加和，在warp内归约，用warp shuffle
    for(int offset = 16; offset > 0; offset >>= 1)
    {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    // 现在 lane 0 有warp的所有元素的 平方和
    if(lane == 0)
    {
        warp_sum[warp_id] = sum;
    }
    __syncthreads();        // 写了共享内存，要屏障

    // 现在所有warp的lane都是 平方和，再次归约
    // 为了更少的屏障，用warp shuffle
    // 一共是8个warp；通过warp_id=0的lane来实现
    if(warp_id == 0)
    {
        float k = (lane < 8) ? warp_sum[lane] : 0.f;
        for(int offset = 4; offset > 0; offset >>= 1)
        {
            k += __shfl_down_sync(0xffffffff, k, offset);
        }
        // 归约后，只有 lane 0 的 k 是正确总和，其他 lane 的 k 是中间值
        // 但 32 个线程都执行 rstd = ...，最后写进去的是谁的值不确定
        // 所以让lane = 0 来写入
        if(lane == 0)
        {
            rstd = rsqrtf( k / cols + eps); 
        }
    }
    __syncthreads();        // rstd是共享内存，要设置屏障
    
    // 现在进行rms归一化，但要从线程的私有数组读取x，不走显存，不走共享内存
    #pragma unroll
    for(int i = 0; i < 16; ++i)
    {
        int offset = blockIdx.x * cols + blockDim.x * i + tid;
        float wv = __half2float(w[blockDim.x * i + tid]);
        out[offset] = __float2half(h[i] * rstd * wv);
    }
}





// kernel1，执行 h = x + res，读x、res，写h，共2048*4096*2/1024/1024 * 3 = 48mb
__global__ void add_residual_kernel(const half* __restrict__ x,
                                    const half* __restrict__ res,
                                    half* __restrict__ h,
                                    const int cols){
    // 2048 * 4096 = 2048 * 256 * 16
    const int tid  = threadIdx.x;
    const int base = blockIdx.x * cols;
    for(int i = tid; i < cols; i += blockDim.x)
    {
        h[base + i] = __float2half(__half2float(x[base + i]) + __half2float(res[base + i]));
    }
}

// kernel2，执行kernel2: out = rmsnorm(h)    读 h, 读w， 写 out
__global__ void rmsnorm_kernel(const half* __restrict__ h,
                                const half* __restrict__ w,
                                half* __restrict__ out,
                                const int cols, float eps){
    // 先读h(x+res) 算平方，归约加和，得到每行的平方和 ——> 算rms
    // 通过warp shuffle 可以减少屏障，一共8个warp，要一个标量来存平方和 ——> lane 0 就有最后warp的平方和，写进共享内存 ——> 归约
    const int tid = threadIdx.x;
    const int lane = threadIdx.x % 32;           // 0-31
    const int warp_id = threadIdx.x / 32;        // 0-7
    float sum_temp = 0.f;
    __shared__ float sum_warp[8];
    __shared__ float rstd;
    for(int i = tid; i < cols; i += blockDim.x)
    {
        // 这里可以问一下，这里读了h，但是没放到共享内存里，还是在显存，等下计算rmsnorm的时候，还要再读一遍h，
        // 可以把 h 放进寄存器，就像融合的方法那样；也可以最开始就拷贝进共享内存里
        float local = __half2float(h[blockIdx.x * cols + i]);
        sum_temp += local * local;
    }
    // warp shuffle
    for(int offset = 16; offset > 0; offset >>= 1)
    {
        sum_temp += __shfl_down_sync(0xffffffff, sum_temp, offset);
    }
    // warp shuffle，不用写屏障
    // 现在要把各个warp的lane 0 的值写到共享内存里面，再进行归约
    if(lane == 0)
    {
        sum_warp[warp_id] = sum_temp;
    }
    // 写了共享内存，要屏障
    __syncthreads();

    // 现在继续归约，得到block的每个元素平方和，采用warp shuffle，减少屏障
    // 归约到warp_id = 0, lane = 0
    if(warp_id == 0)
    {
        float k = (lane < 8) ? sum_warp[lane] : 0.f;
        for(int offset = 4; offset > 0; offset >>= 1)
        {
            k += __shfl_down_sync(0xffffffff, k, offset);
        }
        // for循环结束后，lane0就含有该block的平方总和，直接计算rms，再写入共享内存
        if(lane == 0)
        {
            rstd = rsqrtf(k / cols + eps);
        }
    }
    // 写了共享内存
    __syncthreads();

    // 进行归一化
    for(int i = tid; i < cols; i += blockDim.x)
    {
        const int offset = blockIdx.x * cols; 
        float wv = __half2float(w[i]);
        float hv = __half2float(h[offset + i]);
        out[offset + i] = __float2half(hv * rstd * wv);
    }
}




torch::Tensor fused_rmsnorm_residual(torch::Tensor x, torch::Tensor residual, torch::Tensor w, double eps){
    TORCH_CHECK(x.is_cuda(),"x必须在GPU上");
    TORCH_CHECK(residual.is_cuda(),"residual必须在GPU上");
    TORCH_CHECK(w.is_cuda(),"w必须在GPU上");

    auto xc = x.contiguous();
    auto wc = w.contiguous();
    auto resc = residual.contiguous();
    auto out = torch::empty_like(xc);

    const int cols = (int)xc.size(-1);
    const int rows = (int)xc.numel() / cols;

    const int threads = 256;
    const int blocks = rows;

    fused_rmsnorm_residual_kernel<<<blocks, threads>>>(
        reinterpret_cast<const half*>(xc.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(resc.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(wc.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        rows, cols, (float)eps);

    return out;
}


// host：两次 launch，中间那个 h 就是"分步"的全部代价
torch::Tensor two_step_rmsnorm_residual(torch::Tensor x, torch::Tensor residual,
                                        torch::Tensor w, double eps){
    TORCH_CHECK(x.is_cuda(),"x必须在GPU上");
    TORCH_CHECK(residual.is_cuda(),"residual必须在GPU上");
    TORCH_CHECK(w.is_cuda(),"w必须在GPU上");
    auto xc = x.contiguous();
    auto wc = w.contiguous();
    auto resc = residual.contiguous();
    auto out = torch::empty_like(xc);

    // 还要写中间张量 h
    auto h = torch::empty_like(xc);
    
    const int cols = (int)xc.size(-1);
    const int total = (int)xc.numel();
    const int rows = total / cols;

    const int threads = 256;
    const int blocks = rows;


    add_residual_kernel<<<blocks, threads>>>(
        reinterpret_cast<const half*>(xc.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(resc.data_ptr<at::Half>()),
        reinterpret_cast<half*>(h.data_ptr<at::Half>()),
        cols);


    rmsnorm_kernel<<<blocks, threads>>>(
        reinterpret_cast<const half*>(h.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(wc.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        cols, (float)eps);

    return out;
}