// =============================================================================
//  W2 · RMSNorm —— 起点骨架（阶梯 1-2）
//
//  这个文件现在的状态：
//    ✅ 能编译
//    ✅ 能被 bench_kernels.py 调用
//    ❌ 结果是错的（kernel 里只是把输入拷到输出）
//
//  这是**故意**的。目的：把"工具链问题"和"算法问题"分开。
//  你先跑一次，看到 [WRONG] 就说明工具链没问题了 —— 剩下全是算法。
//
//  你的任务：把下面那个 TODO 实现掉。
//
//  数学定义：
//      rms  = sqrt( mean(x^2) + eps )        对最后一维（4096 个元素）求
//      out  = x / rms * w
//
//  ── 动手前先想清楚这三件事（想不出来就去看《如何写第一个kernel.md》阶梯 3）──
//
//    1) 并行映射：谁负责哪一行？行内的 4096 个元素怎么分配给一个 block 里的线程？
//       （提示：grid.x 对应行号是常见起手式）
//
//    2) 归约：一行 4096 个元素，怎么在一个 block 内求和？
//       这是 CUDA 最经典的练习，也是面试最爱问的。
//
//    3) 两遍还是一遍：先算平方和、再算输出；还是边算边累加？
//       注意 x 会被读几次 —— 这直接决定你的带宽利用率。
//
//  ── 写完之后 ──
//
//    cd ~/projects/aigpu-lab
//    python kernels/bench_kernels.py --only rmsnorm
//
//    目标（W2 验收）：正确性 PASS + 带宽利用率 > 70%（约 190 GB/s）
//    参考：PyTorch 的 layer_norm 是 0.155 ms / 79.8%
// =============================================================================

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// -----------------------------------------------------------------------------
// TODO(你)：实现这个 kernel
//
// 现在它只是"把输入拷到输出"，所以结果必然是错的。
// 这个占位实现的作用只有一个：让整条链路能跑起来。
// -----------------------------------------------------------------------------
__global__ void rmsnorm_kernel(const half* __restrict__ x,
                               const half* __restrict__ w,
                               half* __restrict__ out,
                               const int rows,
                               const int cols,
                               const float eps) {
    const int row = blockIdx.x;      // 一个block负责一行，block内的thread负责
    const int tid = threadIdx.x;     // thread的块内x索引

    // 计算每一行的各线程的平方和
    // x[rows, cols]     输入形状                                   x:   [B, T, 4096]     # 比如 batch × seq_len × hidden
    // w[cols]           权重形状，只作用于最后一维                   w:   [4096]           # 只对最后一维做归一化
    // out[rows, cols]   输出，和输入同形状                          out: [B, T, 4096]     # 和 x 形状相同
    float local = 0.f;
    // 每个线程从自己的 tid 开始，不是0，按 blockDim.x 步长跳跃，这样线程之间不会重叠，合起来覆盖整行
    for(int c = tid; c < cols; c += blockDim.x)
    {
        float b = __half2float(x[row * cols + c]);
        // 每个线程通过tid来累加属于自己要处理的元素
        local += b * b;
    }
    // 声明一个共享内存，大小为 线程数
    // float sdata[cols];    // 错误写法，这是局部数组，不是共享内存
    __shared__ float sdata[256];        // 定义了block=256，一个block有256个线程
    sdata[tid] = local;                 // 每个线程都把自己累加后的值放进去
    // 在 sdata[tid] = local; 之后、归约之前，需要一次同步，保证所有线程都写完了自己的部分和  我忘记写了
    __syncthreads();
    
    // 同步后，sdata[]里面就是一个 block 的 各线程 的累加和，因为一个 block 管理一行，所以进行归约，加起来得到一行的加和
    // 通过归约 计算 每一行的和
    for(int i = blockDim.x / 2; i > 0 ; i >>= 1)            // i >>= 1   i 除以 2 向下取整
    {
        if(tid < i)
        {
            sdata[tid] += sdata[tid + i];
        }
        // 归约结束后，sdata 里仍然有 blockDim.x 个元素，只是 sdata[0] 里存的是总和，其他位置是残留的旧值
        // 每个block都会创建那个共享内存sdata[]，所以每个block规约后，该行的和都在sdata[0]里
        // 这里要进行同步，让同一个 block 里的所有线程，都执行到这一行之后，才继续往下走
        __syncthreads();
    }

    // 现在sdata里面的值就是该block处理的那一行的平方和
    // 算rms，并广播
    // 声明一个共享内存 float 类型 数据
    __shared__ float s_rms;
    if(tid == 0)
    {   
        float mean = sdata[0] / (float)cols;
        s_rms = rsqrtf(mean + eps);                 // rsqrtf() 先开平方，再取倒数
    }
    // 这里需要同步，虽然只有tid=0进行了写操作，但是下面的循环，所有线程都要进行读操作，所以要同步
    __syncthreads();

    // 进行归一化
    // 遍历这一行的 列，读取 x 和 w
    for(int c = tid; c < cols; c += blockDim.x)
    {
        float v = __half2float(x[(long long)row * cols + c]);
        float wv = __half2float(w[c]);
        out[(long long) row * cols + c] = __float2half(v * s_rms * wv);
    }

}

// -----------------------------------------------------------------------------
// 宿主端入口 —— 这部分是样板，一般不用大改
// （只有 grid/block 的配置需要你根据上面的"并行映射"决定后调整）
// -----------------------------------------------------------------------------
torch::Tensor rmsnorm(torch::Tensor x, torch::Tensor w, double eps) {
    TORCH_CHECK(x.is_cuda(), "x 必须在 GPU 上");
    TORCH_CHECK(w.is_cuda(), "w 必须在 GPU 上");
    TORCH_CHECK(x.scalar_type() == torch::kFloat16, "本骨架只支持 fp16");
    TORCH_CHECK(x.dim() >= 1, "x 至少要有 1 维");

    // 如果 x 本来就连续，就返回x本身，不拷贝
    // 否则拷贝一份，变成连续内存
    auto xc = x.contiguous();
    auto wc = w.contiguous();
    auto out = torch::empty_like(xc);

    // xc.size(-1)：取最后一维的大小；
    // 把张量逻辑上展平成 [rows, cols] 的二维矩阵，每一行独立做 RMSNorm
    const int cols = (int)xc.size(-1);
    const int rows = (int)(xc.numel() / cols);

    // TODO(你)：想清楚这里的 grid/block 该怎么配。
    // 当前的配置只保证"每个元素都被覆盖到"，不是按行划分的。
    const int threads = 256;
    // const long long total = (long long)xc.numel();
    // const int blocks = (int)((total + threads - 1) / threads);
    const int blocks = rows;

    rmsnorm_kernel<<<blocks, threads>>>(
        reinterpret_cast<const half*>(xc.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(wc.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        rows, cols, (float)eps);

    return out;
}
