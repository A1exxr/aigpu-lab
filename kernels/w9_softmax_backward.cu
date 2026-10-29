/* 骨架：softmax 反向（w9）
     目标：给定 s = softmax(x) 和 dy，求 dx
     dx_j = s_j * ( dy_j - Σ_i dy_i * s_i )
     访存：读 s(N) + 读 dy(N) + 写 dx(N) = 3N，理论下限 3N/221.4GB/s
*/
#include <cuda_fp16.h>
#include <torch/extension.h>

#define SM_THREADS 256
#define ELEMS_PER_THREAD 16          // cols = 4096 时 256*16 = 4096

__global__ void softmax_backward_kernel(
    const half* __restrict__ s,      // softmax 的输出，[rows, cols]
    const half* __restrict__ dy,     // loss 对 s 的梯度，[rows, cols]
    half* __restrict__ dx,           // 输出，[rows, cols]
    int rows, int cols)
{

    int row = blockIdx.x;                 // 1个block负责1行
    size_t base = (size_t)row * cols;     // 不用int是因为row*col可能超过21亿，size_t是64位无符号类型
    int tid = threadIdx.x;                // 0 1 2 ... 255   256个线程，每个线程负责16个元素，一行有256*16=4096个元素
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x >> 5;
    int warp_num = 256 / 32;


    // 把本线程负责的 16 个元素读进寄存器，
    float dot_partial = 0.f;
    half sreg[16]; 
    half dyreg[16];
    for (int k = 0; k < ELEMS_PER_THREAD; ++k) 
    {
        int j = tid + k * SM_THREADS;        // 跨步，保证合并访存
        sreg[k]  = s[base + j];
        dyreg[k] = dy[base + j];
        dot_partial += __half2float(dyreg[k]) * __half2float(sreg[k]);
    }
   
    

    // 两级归约求出整行的 dot = Σ dot_partial
    for(int offset = 16; offset > 0; offset >>= 1){
        dot_partial += __shfl_down_sync(0xffffffff, dot_partial, offset);
    }

    __shared__ float warp_sum[SM_THREADS / 32];
    __shared__ float dot_shared;

    if(lane == 0)
    {
        warp_sum[warp_id] = dot_partial;
    }
    __syncthreads();

    if(warp_id == 0)
    {
        float k = (lane < warp_num) ? warp_sum[lane] : 0.f;
        for(int offset = warp_num / 2; offset > 0; offset >>= 1)
        {
            k += __shfl_down_sync(0xffffffff, k, offset);
        }
        if(lane == 0)
        {
            dot_shared = k;
        }

    }
    __syncthreads();


    // 用寄存器里已有的 sreg/dyreg 算，不要重读全局
    float dot = dot_shared;
    for(int k = 0; k < ELEMS_PER_THREAD; ++k)
    {
        int j = tid + k * SM_THREADS;
        float sv = __half2float(sreg[k]);
        float dyv = __half2float(dyreg[k]);
        dx[base + j] = __float2half(sv * (dyv - dot));
    }
}

torch::Tensor softmax_backward(torch::Tensor s, torch::Tensor dy) {
    auto rows = s.size(0);
    auto cols = s.size(-1);
    auto dx = torch::empty_like(s);

    softmax_backward_kernel<<<rows, SM_THREADS>>>(
        reinterpret_cast<const half*>(s.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(dy.data_ptr<at::Half>()),
        reinterpret_cast<half*>(dx.data_ptr<at::Half>()),
        (int)rows, (int)cols);

    return dx;
}
