/* ============================================================================
 * w4_attn_naive.cu  —— 骨架（三道 kernel 的调用链 + 内存分配 + 编译入口都搭好了）
 *
 * 三个内层循环故意留空，标了 TODO(你)。你要写的就是那三段。
 *
 * 数据形状（fixed for 本项目）：
 *     Q, K, V : (B, H, S, D)  fp16, 连续
 *     S_mat   : (B, H, S, S)  fp16   ← 故意落显存，为了量 naive 的流量
 *     P       : (B, H, S, S)  fp16   ← softmax 的输出
 *     O       : (B, H, S, D)  fp16
 *
 * 参考：kernels/attention入门-从公式到显存流量.md §2-§4
 * ==========================================================================*/
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <cfloat>

#define SM_THREADS 256
#define MAX_D      64          // 本骨架只支持 D <= 64（Q 行放共享内存）

/* ---------------------------------------------------------------------------
 * kernel 1:  S_mat = Q · Kᵀ / sqrt(D)
 *
 * grid  = B*H*S   （一个 block 负责 S_mat 的一整行，即 (b,h,i) 这一行）
 * block = SM_THREADS
 *
 * 每个 block 的工作：
 *     - 把 Q[b,h,i,:] 这 64 个 half 装进共享内存（128 字节，全 block 共用）,这64个half就是D，每个头要处理的维度
 *     - 每个线程算 S/SM_THREADS 个输出元素 S_mat[b,h,i,j]
 *     - 每个输出元素是一个长度 D 的点积
 * -------------------------------------------------------------------------*/
__global__ void qk_kernel(const half* __restrict__ Q,
                          const half* __restrict__ K,
                          half* __restrict__ S_mat,
                          int S, int D, float scale)
{   
    // 总行数 rows = B * H * S      ———— 0 ~ B*H*S-1
    // row 把 (b,h,i) 三个下标压成一个
    // 其中具体的一行：row = b*H*S + h*S + i 
    //                    = ((b*H + h)*S + i) 
    //                    =  [前面所有 (b,h) 的块] * S + [当前 (b,h) 里的第 i 个]
    // 一个block就负责一个(b, h, i) 组合，在第b个batch，第h个head，序列里第i个位置

    // Q 的一行对应一个token，64维的token向量
    // S_mat 的一行对应一个token，是这个token对所有token的分数

    const int row = blockIdx.x;             // 处理 S_mat 的第 row 行
    const int i   = row % S;                // 这一行对应序列里第 i 个 query token,  i 的范围：0 ~ S-1
    const int tid = threadIdx.x;

    /*
            1个block负责S_mat的一行，其第 i 行 就是 Q · Kᵀ ——> Q[i,:] 与所有 K[j,:]的点积，所以要取Q的第i行 
            S_mat[i, 0] = Q[i,:] · K[0,:] / sqrt(D)
            S_mat[i, 1] = Q[i,:] · K[1,:] / sqrt(D)
            S_mat[i, 2] = Q[i,:] · K[2,:] / sqrt(D)
            ...
            S_mat[i, S-1] = Q[i,:] · K[S-1,:] / sqrt(D)

    */
    // size_t 是无符号的 64 位整数，专门用来表示大小、偏移、下标这类“和内存有关”的量，范围：0 ~ 2^64 - 1（约 1.8 × 10¹⁹）
    // int 上限是 21亿
    // 这一行 Q 的起点，和这个 (b,h) 下 K 的起点
    /*
            Q[b,h,i,:] 的起点 = ((b*H + h)*S + i) * D
                          = row * D

        
        K 要定位的是整个 K 子矩阵的起点（第 0 个 token），不是第 i 个 token：
        K[b,h,0,:] 的起点 = ((b*H + h)*S + 0) * D
                         = ((b*H + h)*S) * D
                         = (row - i) * D
        这里 K 矩阵起点 row - i 是因为， row 本来就表示第i行，而K不从第i行开始，是第0个token开始，所以要减i    
        要算S_mat的第i行，所以Q是第i行，而K要所有的列，
    
    */
    const size_t q_base = (size_t)row * D;                    // Q[b,h,i,0]              取 Q 的第 i 行（64 维）
    const size_t k_base = (size_t)(row - i) * D;              // K[b,h,0,0]              当前 (b,h) 下 K 的起点
    const size_t s_base = (size_t)row * S;                    // S_mat[b,h,i,0]          写 S_mat 的第 i 行（S 个分数）

    __shared__ float q_row[MAX_D];          // Q[i,:] 是 64 维向量,这个 block 内所有线程算点积时，都要反复用它,所以放入共享内存
                                            // [改2] half → float：装配时转一次，后面 1024×64 次循环不再重复转换

    // ---- TODO(你) 1/5：把 Q 行装进 q_row -------------------------------
    // 这里D=64，而线程有256个，是正常的，因为 block 大小不是为加载 Q 行定的，而是为后面的点积计算定的
    // 所以只循环一次，第二次就越界了，直接退出，当blockDim.x < D的时候，才会多次执行
    for(int d = tid; d < min(D, blockDim.x); d+=blockDim.x)   
    {   
        q_row[d] = __half2float(Q[q_base + d]);              
    }
    __syncthreads();        // 每个block都有一个q_row，里面是Q的第i行，来计算S的第i行
    // 两种写法
        // if(tid < D)
        // {
        //     q_row[tid] = q[q_base + tid];
        // }
        // __syncthreads();


    // ---- TODO(你) 2/5：每个线程负责若干 j，算点积 ------------------------
    // blockDim.x 是 0-255，一个block256个线程
    // 外层遍历输出维，内层遍历归约维   [S,D] * [D,S] = [S,S]

    /*
        这里是 一个线程读 K 的一行，进行 64 次串行乘加，算出一个值，放到 S 矩阵里 
        然后 j + 1，第二个线程继续这样算                                          
        线程 0 负责 j = 0, 256, 512, 768     线程 255 负责 j = 255, 511, 767, 1023
        这个方法访存有问题，内层 d 循环每次迭代，这 32 个线程在同一时刻执行同一条 load 指令，但读的地址是：
                                    线程 0:  K[0*64 + d]
                                    线程 1:  K[1*64 + d]
                                    线程 2:  K[2*64 + d]
                                    ...
                                    线程 31: K[31*64 + d]
            地址间隔是 64 个 half = 128 字节。
            对固定的 d，32 个线程的地址落在 32 条不同的 cache line 上，每条 line 只用 2 字节。带宽利用率约 1.5%
            硬件视角：第 0 次迭代，warp 内 32 个线程同时执行 K[j*64 + 0] 这条 load；第 1 次迭代，32 个线程同时执行 K[j*64 + 1]；…… 不是线程 0 先跑完 64 次，线程 1 再跑

        新方法是 一个 warp 算 K 的一行，而是循环处理多个 j                        
    
    */
    for (int j = tid; j < S; j += blockDim.x) {                     
        float acc = 0.f;                                            
        for(int d = 0; d < D; d++)                                  
        {                                                           
            acc += q_row[d] * __half2float(K[k_base + j * D + d]);   
        }
        S_mat[s_base + j] = __float2half(acc * scale);
    }

    // 方法二：一个 warp 算 K 的一行， K 一行有 64个元素，
    const int lane = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;
    
    for(int j = warp_id; j < S; j += 8)
    {
        float acc = 0.f;

        // lane 分摊 d：lane 0 算 d=0,32；lane 1 算 d=1,32 ...
        for(int d = lane; d < D; d += 32)
        {
            acc += q_row[d] * __half2float(K[k_base + j * D + d]);
        }

        // warp 内进行归约，lane 0 得到完整点积
        for(int offset = 16; offset > 0; offset >>= 1)
        {
            acc += __shfl_down_sync(0xffffffff, acc, offset);
        }

        // lane 0 写出结果
        if(lane == 0)
        {
            S_mat[s_base + j] = __float2half(acc * scale);
        }
    }

}

/* ---------------------------------------------------------------------------
 * kernel 2:  按行 softmax（原地，把 S_mat 变成 P）  P是softmax后的注意力分数矩阵[B,H,S,S]
 *
 * grid  = B*H*S   （一个 block 一行，行长 S）
 * block = SM_THREADS
 *
 * 两步：
 *     ① 求这一行的 max   → 用 warp shuffle + 共享内存跨 warp 归约（抄 w2_softmax.cu）
 *     ② 求 sum(exp(x-max))，再原地写回 exp(x-max)/sum
 * 这两步可以直接复刻你 w2_softmax.cu 的结构，只是把"整行"换成"这一行"。
 * -------------------------------------------------------------------------*/
__global__ void softmax_kernel(half* __restrict__ X, int S)
{
    __shared__ float smem[8];        
    const int row = blockIdx.x;
    const size_t base = (size_t)row * S;
    const int tid = threadIdx.x;        
    const int lane = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;
    float local_max = -FLT_MAX;
    float sum = 0.f;
    __shared__ float x_exp[1024];
    __shared__ float x_sum[8];

    // ---- TODO(你) 3/5：求行最大值 --------------------------------------
    // 建议：每线程先在自己负责的那几个 j 上求 local_max（float），
    //       再 warp 内 __shfl_down_sync 归约成每个 warp 一个值，
    //       写进共享内存，再让线程 0 归约成整行 max。
    // 一共256/32=8个warp
    // S_mat [B,H,S,S]  [1024*1024]
    for(int i = tid; i < S; i += blockDim.x)
    {
        local_max = fmaxf(local_max, __half2float(X[base + i]));
    }
    // warp shuffle
    for(int offset = 16; offset > 0; offset >>= 1)
    {
        float tmp = __shfl_down_sync(0xffffffff, local_max, offset);
        local_max = fmaxf(local_max, tmp);
    }
    
    if(lane == 0)
    {
        smem[warp_id] = local_max;
    }
    __syncthreads();
    
    if(warp_id == 0)
    {
        float k = (lane < 8) ? smem[lane] : -FLT_MAX;
        for(int offset = 4; offset > 0; offset >>= 1)
        {
            k = fmaxf(k, __shfl_down_sync(0xffffffff, k, offset));
        }

        if(lane == 0)
        {
            smem[0] = k;
        }
    }
    __syncthreads();        // 现在是smem[0]有每个block的最大值

    // ---- TODO(你) 4/5：求指数和，并原地写回 exp(x-max)/sum -------------
    // 注意：这里必须再遍历一次这一行（softmax 的两趟本质），
    //       第二趟可以直接写 half: X[base+j] = __float2half(expf(v - max) * inv_sum)
    for(int i = tid; i < S; i += blockDim.x)
    {
        float e = expf(__half2float(X[base + i]) - smem[0]);
        x_exp[i] = e;
        sum += e;   // 这里只是线程算了各自的exp 和 
    }

    // warp shuffle 
    for(int offset = 16; offset > 0; offset >>= 1)
    {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    // 现在是lane 0 有warp里面的和
    // 再进行warp的归约，8个warp
    if(lane == 0)
    {
        x_sum[warp_id] = sum;
    }
    __syncthreads();

    if(warp_id == 0)
    {
        float k = (lane < 8) ? x_sum[lane] : 0.f;
        for(int offset = 4; offset > 0; offset >>= 1)
        {
            k += __shfl_down_sync(0xffffffff, k, offset);
        }

        if(lane == 0)
        {
            x_sum[0] = k;
        }
    }
    __syncthreads();

    for(int i = tid; i < S; i += blockDim.x)
    {
        X[base + i] = __float2half(x_exp[i] / x_sum[0]);
    }
}

/* ---------------------------------------------------------------------------
 * kernel 3:  O = P · V
 *
 * grid  = B*H*S   （一个 block 负责 O 的一行，输出 D=64 个数）
 * block = SM_THREADS
 *
 * 一个输出 O[b,h,i,d] = Σ_j P[b,h,i,j] * V[b,h,j,d]     （j 走完整个 S）
 *
 * 两种切法，你自己选一种：
 *   (A) 线程 d 专管一个输出维度：只有 D 个线程在累加，其余闲着（简单，但慢）
 *   (B) 把 j 切成 SM_THREADS/D = 4 段，每段 4 个线程负责同一个 d，
 *       各自累加自己那段 j，最后在共享内存里把这 4 个部分和加起来（快，稍麻烦）
 * 先用 (A) 跑通，数字不对再谈 (B)。
 * -------------------------------------------------------------------------*/
__global__ void pv_kernel(const half* __restrict__ P,
                          const half* __restrict__ V,
                          half* __restrict__ O,
                          int S, int D)
{
    const int row = blockIdx.x;
    const int i   = row % S;
    const size_t p_base = (size_t)row * S;                  // P[b,h,i,0]
    const size_t v_base = (size_t)(row - i) * D;            // V[b,h,0,0]
    const size_t o_base = (size_t)row * D;                  // O[b,h,i,0]
    const int tid = threadIdx.x;

    // ---- TODO(你) 5/5：沿 j 累加 ---------------------------------------
    // 外层遍历输出维，内层遍历归约维  [S,S] * [S,D] = [S,D] 
    for (int d = tid; d < D; d += blockDim.x) 
    {
        float acc = 0.f;
        for (int j = 0; j < S; ++j)
            acc += __half2float(P[p_base + j]) * __half2float(V[v_base + (size_t)j * D + d]);
        O[o_base + d] = __float2half(acc);
    }
    // 先别管访存是否合并，跑通再说；跑通后用 ncu 看这个 kernel 的 DRAM 与 L1。
}

/* ---------------------------------------------------------------------------
 * host：分配中间矩阵 + 依次 launch 三个 kernel
 * -------------------------------------------------------------------------*/
torch::Tensor attn_naive(torch::Tensor Q, torch::Tensor K, torch::Tensor V)
{
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(), "inputs must be CUDA");
    TORCH_CHECK(Q.dtype() == torch::kHalf, "only fp16 supported");
    TORCH_CHECK(Q.dim() == 4, "expected (B,H,S,D)");

    const int B = Q.size(0), H = Q.size(1), S = Q.size(2), D = Q.size(3);
    TORCH_CHECK(D <= MAX_D, "D must be <= 64 in this skeleton");

    auto Qc = Q.contiguous(), Kc = K.contiguous(), Vc = V.contiguous();

    // 中间矩阵：故意落显存（naive 的本质）
    auto opts = torch::TensorOptions().dtype(torch::kHalf).device(Q.device());
    auto S_mat = torch::empty({B, H, S, S}, opts);
    auto O     = torch::empty_like(Qc);

    const int rows  = B * H * S;
    const float scale = 1.0f / sqrtf((float)D);
    const size_t smem = MAX_D * sizeof(half);

    qk_kernel<<<rows, SM_THREADS>>>(
        reinterpret_cast<const half*>(Qc.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(Kc.data_ptr<at::Half>()),
        reinterpret_cast<half*>(S_mat.data_ptr<at::Half>()),
        S, D, scale);

    softmax_kernel<<<rows, SM_THREADS, smem>>>(
        reinterpret_cast<half*>(S_mat.data_ptr<at::Half>()), S);

    pv_kernel<<<rows, SM_THREADS>>>(
        reinterpret_cast<const half*>(S_mat.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(Vc.data_ptr<at::Half>()),
        reinterpret_cast<half*>(O.data_ptr<at::Half>()),
        S, D);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err));

    return O;
}


