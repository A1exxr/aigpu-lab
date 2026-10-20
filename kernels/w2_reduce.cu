// 练习 reduce 归约操作


#include <torch/extension.h>
#include <cuda_runtime.h>


__global__ void reduce_kernel(const float* __restrict__ x,
                                float* __restrict__ out,
                                const int n){
    // n = 4194304, blockDim.x = 512, 每个线程负责8个元素，一个block负责4096个元素，一轮就是512个线程都取一个数，要取4096/512=8轮
    // blockIdx.x: 0 ~ 1023  一共1024个block                    
    int tid = threadIdx.x;    // threadIdx.x: 0 ~ 511  这个只是block内部线程的索引
    float partial = 0.f;
    // __shared__ float s[512];
    // 循环的上界应该是每个 block 负责的元素数；写了好多遍都写不对
    for(int i = tid; i < blockDim.x * 8; i += blockDim.x){
        partial += x[blockIdx.x * blockDim.x * 8 + i];
    }            

    // 这是一般归约方法
    // s[tid] = partial;
    // __syncthreads();
    // for(int i = blockDim.x/2; i > 0; i >>= 1){
    //     if(tid < i)
    //     {
    //         s[tid] += s[tid + i];
    //     }
    //     // 归约的每一层都要同步一下
    //     __syncthreads();
    // }                 
    
    // 下面采用 warp shuffle 在同一个warp内线程直接交换寄存器值，不需要经过共享内存，是warp级别的归约
    // 新结构：
    //      级别 0：每线程把自己那 8 个加起来            → 512 个部分和（每线程 1 个）  网格跨步，每个线程在自己的寄存器里累加，不涉及任何线程间通信，所以没有 CUDA API
    //      级别 1：warp 内 shuffle 归约（32 → 1）        → 16 个部分和（每 warp 1 个）   ★无屏障、无共享内存
    //      级别 2：16 个写进共享内存 → 屏障 → warp 0 再 shuffle 归约  → 1 个总和       ★只有这 1 个屏障，就是正常的归约

    // lane 是warp内线程的编号 0 ~ 31
    //  关于 级别1 的 API，如下：                    第三参数含义                  效果
    //      __shfl_sync(mask, var, srcLane)	        源 lane 的绝对编号	        所有线程从同一个 lane 取值（广播）
    //      __shfl_down_sync(mask, var, delta)	    相对偏移	                线程 lane 从 lane + delta 取值
    //      __shfl_up_sync(mask, var, delta)	    相对偏移	                线程 lane 从 lane - delta 取值
    //      __shfl_xor_sync(mask, var, laneMask)	异或掩码	                线程 lane 从 lane ^ laneMask 取值  
    //  参数    mask：参与的线程掩码，通常 0xffffffff（全部 32 个线程）
    //          var：要传递的变量（可以是 int、float 等），但必须是寄存器里的标量变量
    //          offset：偏移量

    // 上面的partial已经是线程tid的8个元素和了，每个线程都有一个partial
    // 一个warp32个线程
    for(int offset = 16; offset > 0; offset >>= 1)
    {
        // warp内部进行归约，不需要屏障，归约后 每个 warp 的“有效值”只有 lane 0 手里那一个，其余lane的值就无效了
        partial += __shfl_down_sync(0xffffffff, partial, offset);
    }

    // 每个warp都归约了，512/32=16，一共16个warp总和，再次归约，得到block处理的行的总合
    // 但要先写进共享内存，所以现在这个warp_sum[]里面就是所有warp的lane0的值
    // lane就是这样定义的 0~31
    int lane = threadIdx.x % 32;        
    // 同时还要定义warp的id，一个block里面有16个warp
    int warp_id = threadIdx.x / 32;     
    __shared__ float warp_sum[16];
    if(lane == 0)
    {
        warp_sum[warp_id] = partial;
    }
    __syncthreads();        // 写进共享内存，先同步，才能读
    
    // 级别2的归约，16个warp
    for(int i = 16 / 2; i > 0; i >>= 1)
    {   
        // 不能只是这一个warp_id < i 的条件，这只判断了warp的编号，但同一个warp内的warp_id相同，还需要lane来叠加判断
        // 只让每个 warp 的 lane 0 参与
        if(warp_id < i && lane == 0)
        {
            warp_sum[warp_id] += warp_sum[warp_id + i];
        }
        __syncthreads();
    }
    // 现在 warp_sum[0] 就是整个 block 的总和
    // 总和在共享内存中，如果是softmax或别的，就需要写到专门的共享变量里面，让所有线程都可见
    if(warp_id == 0 && lane == 0)
    {
        out[blockIdx.x] = warp_sum[0];
    }



    // // out是[1024,] 形状 ，对应 1024 个block
    // if(tid == 0)
    // {
    //     out[blockIdx.x] = s[tid];
    // }
        
}






torch::Tensor reduce_sum(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x 必须在 GPU 上");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "只支持 fp32");
    auto xc = x.contiguous();
    const int n = (int)xc.numel();

    auto out = torch::empty({1024}, x.options());      // 1024 个 block 各写一个部分和

    reduce_kernel<<<1024, 512>>>(xc.data_ptr<float>(), out.data_ptr<float>(), n);

    return out.sum();                                   // 1024 个数 → 1 个数（PyTorch 代劳）
}