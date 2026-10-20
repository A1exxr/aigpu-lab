// softmax 练习

#include <torch/extension.h>
#include <cuda_runtime.h>

// fp16 要这个依赖
#include <cuda_fp16.h>

// -inf 
#include <cfloat>


// 输入输出形状是 [1, 2048, 4096]   fp16
// 但这里不需要管第一维 就是 [2048, 4096]    8,388,608

'''
    现在写完kernel了， 求max和exp全局读了两次，输出out时全局写了一次 ————> 把两次读变成一次
    看完这个后，看优化后的版本
'''


__global__ void softmax_kernel(const half* __restrict__ x,
                                half* __restrict__ out,
                                const int rows, const int cols){
    // 思路：找到每行最大值，每行所有元素减去最大值 ——> 所有元素取e ——> 每行加和 ——> 各元素/和
    // 一个block处理一行，4096个元素，一共2048行，所以会溢出，block也要循环，还要检查边界
    // 现在先不考虑这些，还是在1维网格里，一个block处理一行，设置 2048 个block                            
    const int tid = threadIdx.x;        // 0 ~ 511        一个thread处理8个元素
    // 设置共享内存保存最大值
    __shared__ float x_max[512];
    if(tid < 512){
        // 归约的初值，不能用0，要用-infinite
        // x_max[tid] = 0.f;
        x_max[tid] = -FLT_MAX;
    }
    __syncthreads();

    
    // 找单个线程处理元素中的最大值
    for(int i = tid; i < cols; i += blockDim.x)
    {   
        // 这时候最大值是 某个线程处理的所有元素中的 最大值，不是一个block的，block得所有线程一起比
        // x[blockIdx.x * row + i] 不能是*row，这算出来是 blockIdx.x²
        // blockIdx.x 乘的是「一个 block 覆盖多少个元素」= cols
        // 一个half类型， 一个float类型， 不能直接比大小，编译器不认识，要显示转换
        float v = __half2float(x[blockIdx.x * cols + i]);
        if(x_max[tid] < v)
        {
            x_max[tid] = v;
        }
    }
    __syncthreads();

    // 要通过归约找最大值？
    for(int i = blockDim.x / 2; i > 0; i >>= 1)
    {   
        if(tid < i)
        {
            if(x_max[tid] < x_max[tid + i])
            {
                // 不能用 == 这是比较，不是赋值
                // x_max[tid] == x_max[tid + i];
                x_max[tid] = x_max[tid + i];
            }
        }
        __syncthreads();
    }
    // 此时max[0]就是每行的最大值了，让每个数都减去该行最大值
    // sum要进行初始化；这是线程私有的，每个线程都有一个sum 
    float sum = 0.f;
    
    // 这里全错
    // for(int i = tid; i < cols; i += blockDim.x)
    // {   
    //     // 不能把exp结果写回x里，x是输入变量，不然运行完程序就找不到输入了，得用内存来存放exp，共享内存或全局内存
    //     x[blockIdx.x * cols + i] -= max[0];
    //     // 在以e为底
    //     x[blockIdx.x * cols + i] = expf(x[blockIdx.x * cols + i]);
    //     // 每行加和
    //     sum += x[blockIdx.x * row + i];
    // }
    // __syncthreads();

    // 加和要进行归约，我没做，我没想到
    // 要先决定exp放哪里，共享内存要多少字节？  放全局内存要增加多少读写操作，带宽上限还剩多少？
    __shared__ float x_exp[4096];         // 一行是4096个元素
    for(int i = tid; i < cols; i += blockDim.x)
    {
        // 先减去最大值，再以e为底
        // expf里面 前面x[]是half，后面x_max[]是half
        // 转换一下
        float e = expf(__half2float(x[blockIdx.x * cols + i]) - x_max[0]);
        x_exp[i] = e;   
        // 每个线程都把自己的sum算出来了，然后归约加和得到block这一行的sum
        sum += e;
    }
    // __syncthreads();         // 这个是多余的，因为exp是本线程写，本线程读，不需要屏障

    // 刚刚求了每个线程 处理元素的和，把这个和 存进共享内存里
    __shared__ float sum_data[512];     
    sum_data[tid] = sum;
    // 这里要一个屏障，我没写 ————  写完共享内存 → __syncthreads() → 才能跨线程读。
    __syncthreads();

    // 归约加和
    for(int i = blockDim.x/2; i > 0; i >>= 1)
    {
        if(tid < i)
        {
            sum_data[tid] += sum_data[tid + i];
        }
        __syncthreads();   // 现在sum_data[0]就是每行的和
    }

    // 进行归一化
    for(int i = tid; i < cols; i += blockDim.x)
    {
        // 右边是float，左边out是fp16类型，要进行转换
        out[blockIdx.x * cols + i] = __float2half(x_exp[i] / sum_data[0]);
    }
}



torch::Tensor softmax(torch::Tensor x){
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