#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cfloat>
#include <cstdlib>      // ← 新增（getenv / atoi）

// 这个版本是 要把两次归约 改成warp shuffle
// 一次是求max，一次是求和
// 先搬一次数据，把所有x都搬进共享内存smem里面，一个block一个共享内存：4096*4 bytes （x是half类型，可以存half类型数据）
// 第一次归约找最大值：级别0，tid线程把8个数据先比大小，得到tid的最大值，才能到warp里面去比，
//                   级别1，warp内比大小，最大的放到lane=0 ——> 每个warp都是lane=0，每个warp都要放个中间值到x_max里； 16*4 bytes
//                   级别2，16个warp的lane=0进行归约比较，warp_id=0且lane=0是该block的最大值 ——> x_max[0]

// 所有元素要减去该行最大值，再都以e为底，然后存到到exp[]共享内存里  大小为 4096*4 bytes 或者直接覆盖到 smem里面？

// 第二次归约求和：级别0，显示线程把自己的8个数加起来 ——> 512个线程
//                级别1，线程间直接传递信息，warp内归约，得到16个部分和，是中间值,sum_data[] 大小为 16*4 bytes
//                级别2，将16个warp部分和归约得到1个block的和；这个和在sum_data[0]上
// 两次归约结束了，然后再计算，输出
// 注意，两次归约后，都需要进行广播

// 这样两次归约，级别1都是warp shuffle；级别2是共享内存树形归约 16→8→4→2→1
// 再次优化，把两个归约的级别2，也写成warp shuffle
// 16个 warp 的 lane=0 各自把结果写进x_max[]里，再让warp0的前16个线程读数据，后16个线程设置为-inf，然后warp0内部进行归约


__global__ void softmax_kernel(const half* __restrict__ x,
                               half* __restrict__ out,
                               const int rows, const int cols){
    const int tid = threadIdx.x;            // 0-511
    const int lane = threadIdx.x % 32;          // 0-31
    const int warp_id = threadIdx.x / 32;       // 0-15

    // 如果用half存数据，使用共享内存更小了  4096*2 bytes
    extern __shared__ char smem[];                               // 用 char* 做基址，方便按字节偏移
    // half* x_smem = reinterpret_cast<half*>(smem);
    float* x_exp = reinterpret_cast<float*>(smem);
    float* x_max = reinterpret_cast<float*>(smem + cols * 4);
    float* sum_data = reinterpret_cast<float*>(smem + cols * 4 + 16 * 4);

    // ---------------------------------------------------------------
    // ① 搬进共享内存 —— 全 kernel 【唯一一次】全局读
    // ---------------------------------------------------------------

    for(int i = tid; i < cols; i += blockDim.x)
    {
        x_exp[i] = __half2float(x[blockIdx.x * cols + i]);
    }

    // 求最大值 级别0
    float local_max = -FLT_MAX;
    for(int i = tid; i < cols; i += blockDim.x)
    {
        local_max = fmaxf(local_max, x_exp[i]);     
    }
    // 现在local_max是每个线程处理的8个元素中的最大值，每个线程私有的

    // 级别1，直接在warp里面比大小，不通过tid要处理的8个元素比大小
    for(int offset = 16; offset > 0; offset >>= 1)
    {
        // 当前线程 lane 从 lane + offset 那个线程拿到的 local_max 值。
        float other = __shfl_down_sync(0xffffffff, local_max, offset);
        local_max = fmaxf(local_max, other);    
    }
    // 此时 warp_id 的 lane=0 有最大值，这是warp内部归约，不用屏障
    // 这时候要广播，不然没东西比，把各个warp的最大值存进去
    if(lane == 0)
    {
        x_max[warp_id] = local_max;
    }
    __syncthreads();        // 此时x_max[0,1,2,3,...,15]的值就是对应warp_id的最大值
    
    // // 级别2，将16个warp的lane0进行归约 ——> warp 0 且 lane 0 有block的最大值
    // for(int i = 16 / 2; i > 0; i >>= 1)
    // {
    //     if(warp_id < i && lane == 0)
    //     {
    //         x_max[warp_id] = fmaxf(x_max[warp_id], x_max[warp_id + i]);
    //     }
    //     __syncthreads();
    // }
    // // 此时x_max[0]里面是每行的最大值

    // 级别2，改为warp shuffle归约
    // lane=0 的值 已经通过广播 写进了x_max[]里面
    // 让warp 0 来读这些值，通过lane来实现的， lane是通过threadIdx.x % 32 定义的，属于每个线程私有的，跨warp的线程可能会有一样的lane，但同一个线程束就不会
    // 这里限定了warp_ip=0，在这个线程束里，每个线程的lane都不一样
    if(warp_id == 0)
    {   
        // 这样就把前16个线程选出来了，这里每个线程的lane都不一样
        // 这里temp也是线程私有的值，在寄存器里面
        float temp = (lane < 16) ? x_max[lane] : -FLT_MAX; 
        // 直接进行warp shuffle归约，选最大值
        for(int offset = 16; offset > 0; offset >>= 1)
        {
            float m = __shfl_down_sync(0xffffffff, temp, offset);
            // x_max[lane] = fmaxf(x_max[lane], m);   这样写是错的，不能在共享内存里面归约，要在寄存器上
            temp = fmaxf(temp, m);      // 要这样写
            // 然后再把temp写进共享内存里
        }   
        // 要在if里面写，temp是局部变量
        if(lane == 0)
        {
            x_max[0] = temp;
        }
    }
    __syncthreads();


    // 第二次归约，加和，级别0
    float sum = 0.f;
    for(int i = tid; i < cols; i += blockDim.x)
    {
        float e = expf(x_exp[i] - x_max[0]);
        x_exp[i] = e;                 // 原始值在这里被覆盖，不再需要
        sum += e;
    }

    // 级别1，进行warp内部的归约，加和
    for(int offset = 16; offset > 0; offset >>= 1)
    {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    // 不用同步；sum就是这个warp的元素和
    // 进行广播
    if(lane == 0)
    {
        sum_data[warp_id] = sum;
    }
    __syncthreads();

    // // 级别2，16个warp归约，加和
    // for(int i = 16/2; i > 0; i >>= 1)
    // {
    //     if(warp_id < i && lane == 0)
    //     {
    //         sum_data[warp_id] += sum_data[warp_id + i];
    //     }
    //     __syncthreads();
    // }
    
    // 级别2，优化，写成warp shuffle
    // 也已经广播了，sum_data[]里面的值就是warp的内部和
    // 让warp 0 来读
    if(warp_id == 0)
    {
        float temp = (lane < 16) ? sum_data[lane] : 0;
        // 进行归约
        for(int offset = 16; offset > 0; offset >>= 1)
        {
            // sum_data[lane] += __shfl_down_sync(0xffffffff, temp, offset);  // 所以这里也错了，不能在共享内存上归约
            temp += __shfl_down_sync(0xffffffff, temp, offset);
        }
        if(lane == 0)
        {
            sum_data[0] = temp;
        }
    }
    __syncthreads();


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

    const size_t smem_bytes = (size_t)(cols + 32) * sizeof(float);

    softmax_kernel<<<blocks, threads, smem_bytes>>>(
        reinterpret_cast<const half*>(xc.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        rows,cols
    );
    return out;
}