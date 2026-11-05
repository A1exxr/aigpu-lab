// cpasync_dblbuf.cu —— 双缓冲软件流水线（第一件：不用 cp.async）
//
// 目的：只练“两块 smem + 换顺序 + 两个屏障”这三件事。
// 数据：4 MB（NCHUNK=1024 块，每块 CHUNK=1024 个 float）
// 单 block、256 线程 → 每个线程每块处理 4 个元素。
//
// 编译/运行（WSL）：
//   cd ~/projects/aigpu-lab/kernels
//   ~/miniconda3/envs/triton/bin/python cpasync_dblbuf.py   # 若用 torch load_inline
//   或者：nvcc -O3 -arch=sm_89 cpasync_dblbuf.cu -o /tmp/dblbuf && /tmp/dblbuf
//
// 你要写的只有两个 kernel 的函数体（标了 TODO）。host 部分已经写好，负责
// 分配、计时、与 CPU 串行结果对比——你不需要动它。

#include <cstdio>
#include <cuda_runtime.h>
#include <cmath>

#define CHUNK     1024          // 每块多少个 float
#define NCHUNK    1024          // 多少块  → 总共 4 MB
#define THREADS   256
#define NELEM     (CHUNK * NCHUNK)   // 别叫 N：cuda_pipeline.h 里有 template<unsigned N>

// ---------------------------------------------------------------------------
// 基线：单缓冲，搬完再算
// ---------------------------------------------------------------------------
__global__ void sum_serial(const float* __restrict__ in, float* out, int nchunk) {
    __shared__ float buf[CHUNK];
    float acc = 0.0f;

    for(int c = 0; c < NCHUNK; ++c)
    {   
        // 256线程，每个线程搬4个float
        for(int i = threadIdx.x; i < 1024; i += blockDim.x)
        {
            buf[i] = in[CHUNK * c  + i];
        }
        __syncthreads();

        // 在块内就计算
        for(int i = threadIdx.x; i < 1024; i += blockDim.x)
        {
            acc += buf[i] * 2.0f;
        }
        __syncthreads();
    }

    for (int offset = 16; offset > 0; offset >>= 1) 
    {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }
    __shared__ float warp_sum[THREADS / 32];
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) warp_sum[wid] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.0f;
        for (int w = 0; w < THREADS / 32; ++w) s += warp_sum[w];
        out[0] = s;
    }
}

// ---------------------------------------------------------------------------
// 双缓冲：边搬边算
// ---------------------------------------------------------------------------
__global__ void sum_double_buf(const float* __restrict__ in, float* out, int nchunk) {
    __shared__ float buf[2][CHUNK];     // 两块
    float acc = 0.0f;

    // 预取第 0 块到 buf[0]
    // 块内按 stride 搬 in[0*CHUNK + i] → buf[0][i]
    for(int i = threadIdx.x; i < CHUNK; i += blockDim.x)
    {
        buf[0][i] = in[0*CHUNK + i];
    }
    __syncthreads();

    // 外层循环 c 从 0 到 nchunk-1，循环体里依次做：
    for(int c = 0; c < NCHUNK; ++c)
    {
        // 有1024个块，buf只有两块，所以第c块放到哪个smem是一个按2循环的映射
        int cur = c % 2;
        int nxt = 1 - cur;
        if(c + 1 < nchunk)
        {
            for(int i = threadIdx.x; i < CHUNK; i += blockDim.x)
            {
                buf[nxt][i] = in[(c + 1) * CHUNK + i];
            }            
        }
        __syncthreads();
        for(int i = threadIdx.x; i < CHUNK; i += blockDim.x)
        {
            acc += buf[cur][i] * 2.0f;
        }
        __syncthreads();
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }
    __shared__ float warp_sum[THREADS / 32];
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) warp_sum[wid] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.0f;
        for (int w = 0; w < THREADS / 32; ++w) s += warp_sum[w];
        out[0] = s;
    }
}


// __pipeline_*  让搬运不经过寄存器
#include<cuda_pipeline.h>
__global__ void sum_cp_async(const float* __restrict__ in, float* out, int nchunk)
{
    // smem 16字节对齐
    __shared__ __align__(16) float buf[2][CHUNK];
    float acc = 0.f;
    // 预取第0块
    for(int i = threadIdx.x * 4; i < CHUNK; i += blockDim.x * 4)
    {
        // 循环只管拷贝，其他三行在外面
        // __pipeline_* 的队列是每个线程私有的，不是每个 block 共享的
        // memcpy_async = 往“我的队列”里塞一条拷贝
        // 所以这里是每个线程都去搬自己的4个float，16字节
        // 参1：目标地址；参2：源地址；参3：从源地址开始，要搬运的字节数，同时指定对齐要求
        __pipeline_memcpy_async(&buf[0][i], &in[i], 16);    // 一次异步拷贝4个float
    }
    __pipeline_commit();                // commit = 把“我目前塞进去的这些”封成一组；
                                        // wait_prior(n) = 等“我的”组，直到只剩最近 n 组没完成。
    __pipeline_wait_prior(0);           // 只要这一组，全等
    __syncthreads();

    for(int c = 0; c < nchunk; ++c)
    {
        int cur = c % 2;
        int nxt = 1 - cur;
        if(c + 1 < nchunk)
        {
            for(int i = threadIdx.x * 4; i < CHUNK; i += blockDim.x * 4)
            {
                __pipeline_memcpy_async(&buf[nxt][i], &in[(c+1)*CHUNK + i], 16);
            }
            __pipeline_commit();
        }
        // 最后一轮时，没有新commit，未完成组只剩下NCHUNK-1块这一组，这时wait_prior(1)会认为它可以不等，读到没搬完的数据，结果错可能不报错
        __pipeline_wait_prior((c+ 1 < nchunk) ? 1 : 0);
        __syncthreads();

        for(int i = threadIdx.x; i < CHUNK; i += blockDim.x)
        {
            acc += buf[cur][i] * 2.0f;
        }
        __syncthreads();
    }

    for(int offset = 16; offset > 0; offset >>= 1)
    {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }

    __shared__ float warp_sum[THREADS/32];
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    if(lane == 0)
    {
        warp_sum[warp_id] = acc; 
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.0f;
        for (int w = 0; w < THREADS / 32; ++w) s += warp_sum[w];
        out[0] = s;
    }
}


// ---------------------------------------------------------------------------
// host：已写好，不要改
// ---------------------------------------------------------------------------
static float bench(void (*launch)(const float*, float*, int), const float* d_in,
                   float* d_out, int iters) {
    // 单 block 的微基准会被“显卡升降频”严重干扰：同一份代码两次能差 40%。
    // 对策：跑 ROUND 轮，每轮 iters 次，取**最小值**（最小值最接近“没有干扰”）。
    const int ROUND = 5;
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    for (int i = 0; i < 200; ++i) launch(d_in, d_out, NCHUNK);   // 预热，把频率拉起来
    float best = 1e30f;
    for (int r = 0; r < ROUND; ++r) {
        cudaEventRecord(a);
        for (int i = 0; i < iters; ++i) launch(d_in, d_out, NCHUNK);
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, a, b);
        if (ms / iters < best) best = ms / iters;
    }
    cudaEventDestroy(a); cudaEventDestroy(b);
    return best;
}

static void launch_serial(const float* in, float* out, int nchunk) {
    sum_serial<<<1, THREADS>>>(in, out, nchunk);
}
static void launch_double(const float* in, float* out, int nchunk) {
    sum_double_buf<<<1, THREADS>>>(in, out, nchunk);
}
static void launch_cp_async(const float* in, float* out, int nchunk) {
    sum_cp_async<<<1, THREADS>>>(in, out, nchunk);
}

int main() {
    size_t bytes = (size_t)NELEM * sizeof(float);
    float* h_in = (float*)malloc(bytes);
    // 数据必须有变化：全 1 会让“读错位置”这类 bug 完全看不出来
    for (int i = 0; i < NELEM; ++i) h_in[i] = (float)(i % 97) + 0.5f;

    float* d_in = nullptr; float* d_out = nullptr;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, sizeof(float));
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    // CPU 标准答案：逐元素 *2 累加（用 double，避免求和顺序带来的误差）
    double expect = 0.0;
    for (int i = 0; i < NELEM; ++i) expect += (double)h_in[i] * 2.0;

    float out1 = 0.0f, out2 = 0.0f;
    launch_serial(d_in, d_out, NCHUNK);
    cudaMemcpy(&out1, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    launch_double(d_in, d_out, NCHUNK);
    cudaMemcpy(&out2, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    float out3 = 0.0f;
    launch_cp_async(d_in, d_out, NCHUNK);
    cudaMemcpy(&out3, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    printf("期望 %.1f  单缓冲 %.1f  双缓冲 %.1f  cp.async %.1f\n", expect, out1, out2, out3);
    // 浮点求和必然有微小误差，用相对误差判定
    double e1 = fabs((double)out1 - expect) / expect;
    double e2 = fabs((double)out2 - expect) / expect;
    double e3 = fabs((double)out3 - expect) / expect;
    printf("校验：单缓冲 %s (rel %.2e)  双缓冲 %s (rel %.2e)  cp.async %s (rel %.2e)\n",
           (e1 < 1e-5) ? "OK" : "WRONG", e1,
           (e2 < 1e-5) ? "OK" : "WRONG", e2,
           (e3 < 1e-5) ? "OK" : "WRONG", e3);

    float t1 = bench(launch_serial, d_in, d_out, 200);
    float t2 = bench(launch_double, d_in, d_out, 200);
    float t3 = bench(launch_cp_async, d_in, d_out, 200);

    printf("\n%-24s %10s %8s\n", "版本", "时间 ms", "相对");
    printf("%-24s %10.4f %8.2f\n", "单缓冲（搬完再算）", t1, 1.00);
    printf("%-24s %10.4f %8.2f\n", "双缓冲（边搬边算）", t2, t2 / t1);
    printf("%-24s %10.4f %8.2f\n", "cp.async 双缓冲", t3, t3 / t1);

    cudaFree(d_in); cudaFree(d_out); free(h_in);
    return 0;
}
