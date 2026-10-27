// 实验 A/B/C 共用的探针 kernel

#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void empty_kernel() {
}

__global__ void write1_kernel(float* out) {
    if (blockIdx.x == 0 && threadIdx.x == 0) out[0] += 1.0f;
}

// 空转 kernel：用来演示"两个 kernel 真能同时跑"
// 一个 SM 一个 block，占 1 个 SM，其它 SM 全空着 → 4 个并发能分到 4 个 SM
__global__ void spin_kernel(long long cycles) {
    long long t0 = clock64();
    while (clock64() - t0 < cycles) {
    }
}


// ===== host wrapper =====

void empty_launch() {
    empty_kernel<<<1, 32>>>();
}

// 指定流 —— 捕获 CUDA Graph / 做并发实验都必须用它
void empty_launch_on(int64_t stream_ptr) {
    empty_kernel<<<1, 32, 0, (cudaStream_t)stream_ptr>>>();
}

void spin_launch(int64_t cycles, int64_t stream_ptr) {
    spin_kernel<<<1, 32, 0, (cudaStream_t)stream_ptr>>>((long long)cycles);
}

void write1_launch(torch::Tensor out) {
    write1_kernel<<<1, 32>>>(out.data_ptr<float>());
}
