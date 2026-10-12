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
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long n = (long long)rows * cols;
    if (i < n) {
        out[i] = x[i];          // ← 占位。你要把这行换成真正的 RMSNorm
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

    auto xc = x.contiguous();
    auto wc = w.contiguous();
    auto out = torch::empty_like(xc);

    const int cols = (int)xc.size(-1);
    const int rows = (int)(xc.numel() / cols);

    // TODO(你)：想清楚这里的 grid/block 该怎么配。
    // 当前的配置只保证"每个元素都被覆盖到"，不是按行划分的。
    const int threads = 256;
    const long long total = (long long)xc.numel();
    const int blocks = (int)((total + threads - 1) / threads);

    rmsnorm_kernel<<<blocks, threads>>>(
        reinterpret_cast<const half*>(xc.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(wc.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        rows, cols, (float)eps);

    return out;
}
