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
    // 方法一
    // for (int j = tid; j < S; j += blockDim.x) {                     
    //     float acc = 0.f;                                            
    //     for(int d = 0; d < D; d++)                                  
    //     {                                                           
    //         acc += q_row[d] * __half2float(K[k_base + j * D + d]);   
    //     }
    //     S_mat[s_base + j] = __float2half(acc * scale);
    // }

    // 方法二：一个 warp 读这 K 的 64 个元素，和 q_row 做点积，归约后得到 S_mat 的一个元素（1 个标量）
    // 但这还是有问题，每一个 warp 只能算一个 S_mat元素，中间要经过5次 shuffle，这5次是串行的，第2轮必须等第1轮的结果
    // 硬件手上没有别的活可发，只能干等那 20–30 个周期。这就是算出来的 16% 发射效率。
    const int lane = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;
    const int warp_nums = blockDim.x / 32;
    
        // for(int j = warp_id; j < S; j += warp_nums)
        // {
        //     float acc = 0.f;

        //     // lane 分摊 d：lane 0 算 d=0,32；lane 1 算 d=1,32 ...
        //     for(int d = lane; d < D; d += 32)
        //     {
        //         acc += q_row[d] * __half2float(K[k_base + j * D + d]);
        //     }

        //     // warp 内进行归约，lane 0 得到完整点积
        //     for(int offset = 16; offset > 0; offset >>= 1)
        //     {
        //         acc += __shfl_down_sync(0xffffffff, acc, offset);
        //     }

        //     // lane 0 写出结果
        //     if(lane == 0)
        //     {
        //         S_mat[s_base + j] = __float2half(acc * scale);
        //     }
        // }

    // 方法三：ILP-4 = 让一个线程同时算 4 个相邻的 S 元素，而不是 1 个。
    // ILP-4 不是 lane 分组，而是“同一个 lane 同时算 4 个 j”，32 个 lane 仍然全部参与每个 j 的计算和归约。  让lane分组是另一种优化方法
    // 一次维护 4 个累加器，让 4 条 shuffle 链并行
    /*
        在 warp 视角，S 有 1024个j， 8 个warp分摊，1个warp负责128个j，每次处理4个j，一共循环128/4=32次

        在 block 视角，8个warp同时跑，同时一起处理4*8=32个j，1024/32=32，block也要循环32次
    
    */
    for(int j_base = warp_id * 4; j_base < S; j_base += warp_nums * 4)          // warp_id 0-7     warp_nums = 8，
    {
        int j0 = j_base, j1 = j_base + 1, j2 = j_base + 2, j3 = j_base + 3;     // 4个j块，连续的 
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;       // 4个累加器，每个线程都有这4个累加器，warp shuffle归约时，就根据累加器来归约的
        for(int d = lane; d < D; d += 32)
        {
            float q = q_row[d];
            a0 += q * __half2float(K[k_base + j0 * D + d]);
            a1 += q * __half2float(K[k_base + j1 * D + d]);
            a2 += q * __half2float(K[k_base + j2 * D + d]);
            a3 += q * __half2float(K[k_base + j3 * D + d]);
        }

        for(int offset = 16; offset > 0; offset >>= 1)
        {
            a0 += __shfl_down_sync(0xffffffff, a0, offset);          // 4 组 shuffle 交错，每轮 4 条独立指令
            a1 += __shfl_down_sync(0xffffffff, a1, offset);
            a2 += __shfl_down_sync(0xffffffff, a2, offset);
            a3 += __shfl_down_sync(0xffffffff, a3, offset);
        }

        if(lane == 0)
        {
            S_mat[s_base + j0] = __float2half(a0 * scale); 
            S_mat[s_base + j1] = __float2half(a1 * scale);
            S_mat[s_base + j2] = __float2half(a2 * scale);
            S_mat[s_base + j3] = __float2half(a3 * scale);
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
        // for (int d = tid; d < D; d += blockDim.x)                // 外层 d = tid → 只有 tid 0..63 有活干（192 个线程直接跳过循环）
        // {                                                        // 每个干活的线程串行扫 1024 个 j，每次 1 个 P + 1 个 V 的 load
        //     float acc = 0.f;
        //     for (int j = 0; j < S; ++j)
        //         acc += __half2float(P[p_base + j]) * __half2float(V[v_base + (size_t)j * D + d]);
        //     O[o_base + d] = __float2half(acc);
        // }
    // 先别管访存是否合并，跑通再说；跑通后用 ncu 看这个 kernel 的 DRAM 与 L1。
    // 现在改 P @ V —— 256个线程分4组，j也分4组，每个线程扫 256 个j，少了4倍串行，但还是在一个block里
    // 切分的就是P矩阵的行、V矩阵的列 ———— j 轴(P的列方向 = V的行方向)，64 个线程一组，切完各自算部分和，最后加起来，每段算的是「部分乘加」
    // O[d] = partial[0][d] + partial[1][d] + partial[2][d] + partial[3][d]
    const int group_nums = blockDim.x / D;              // 组数，256/64 = 4
    const int group_id = threadIdx.x / D;               // 当前组编号，0 1 2 3
    const int d = threadIdx.x % D;                      // 线程负责的维度，0 1 2 ... 63，共64个维度，256个线程 ———— 4个线程处理一个d

    // 这组负责的 j 区间    
    const int seg = S / group_nums;                 // 1024/4 = 256     一个区间有256个元素要处理，一个线程固定一个 d，但一个 d 由 4 个线程协作完成
    const int j0 = group_id * seg;
    const int j1 = j0 + seg;                        // 每组处理的区间 j0 —— j0 + seg，就是一个 seg，4个seg就覆盖了一个S

    float acc = 0.f;
    for(int j = j0; j < j1; ++j)
    {
        acc += __half2float(P[p_base + j]) * __half2float(V[v_base + j * D + d]);
    }
    // 这里一共是4 个组，每组 64 个线程，共256个线程，算了 4 组 × 64 个 d = 256 个部分和，不用在warp里归约，写进共享内存，按线程编号存储；
    __shared__ float partial[SM_THREADS];       
    partial[tid] = acc;
    __syncthreads();

    // 前D个线程负责合并这4段partial，因为1个线程处理1个d，所以只需要D个线程
    if(threadIdx.x < D)
    {
        float sum = 0.f;
        for(int g = 0; g < group_nums; ++g)
        {
            sum += partial[g * D + threadIdx.x];        //  第 g 组算的第 d 列的部分和，前D个线程的id和d列是对应的
        }
        O[o_base + threadIdx.x] = __float2half(sum);
    }
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


