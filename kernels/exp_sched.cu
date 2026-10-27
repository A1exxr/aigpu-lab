// exp_sched.cu —— GPU 调度层三个实验，纯 C++/CUDA，不经过 Python
//
// 为什么用 C++：Python 每次调 kernel 都要付 pybind 绑定开销（约 6 µs），
// 会把这个实验要测的“launch 成本”淹没。C++ 里提交才是真实引擎的成本。
//
// 用法：
//   nvcc -O3 -std=c++17 -o exp_sched exp_sched.cu
//   ./exp_sched
//
// 四个实验：
//   A  launch 成本：1000 次空 kernel 连发
//   B  多流：4 条流各发 250 次（空 kernel）
//   C  CUDA Graph：N = 1,2,4,8,16,32,64 个节点，测重放时间
//   D  真并发：4 个 spin kernel 串行 vs 4 条流并发
//
// 判据：
//   A 给“每次提交要多少钱”；C 给“图内每次只要多少钱”；两者之差就是 graph 的天花板收益。
//   B 的空 kernel 若比 A 快，说明省的 CPU 提交流水线（不是 GPU 并行）；
//   D 才是 GPU 真并行，因为每个 spin kernel 只占 1 个 SM，24 SM 有 23 个空着。

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <algorithm>
#include <functional>

#define CK(x)                                                                        \
    do {                                                                             \
        cudaError_t err_ = (x);                                                      \
        if (err_ != cudaSuccess) {                                                   \
            printf("[CUDA ERROR] %s  (%s:%d)\n", cudaGetErrorString(err_),           \
                   __FILE__, __LINE__);                                              \
            std::exit(1);                                                            \
        }                                                                            \
    } while (0)


__global__ void empty_kernel() {}

__global__ void spin_kernel(long long cycles) {
    long long t0 = clock64();
    while (clock64() - t0 < cycles) {
    }
}


// 一次测量：event 包住 f，返回 µs
static double time_us(const std::function<void()>& f) {
    cudaEvent_t s, e;
    CK(cudaEventCreate(&s));
    CK(cudaEventCreate(&e));
    CK(cudaEventRecord(s));
    f();
    CK(cudaEventRecord(e));
    CK(cudaEventSynchronize(e));
    float ms = 0.0f;
    CK(cudaEventElapsedTime(&ms, s, e));
    CK(cudaEventDestroy(s));
    CK(cudaEventDestroy(e));
    return ms * 1000.0;
}

// 重复 rounds 次取最小：CPU 侧噪声只会让数字变大，最小值最接近真实下限
static double min_us(const std::function<void()>& f, int rounds = 9, int warmup = 20) {
    for (int i = 0; i < warmup; ++i) f();
    CK(cudaDeviceSynchronize());
    double best = 1e18;
    for (int i = 0; i < rounds; ++i) best = std::min(best, time_us(f));
    return best;
}


// A：1000 次连发（全部提交完再同步）
static double bench_serial_launch(int n = 1000) {
    return min_us([n] {
        for (int i = 0; i < n; ++i) empty_kernel<<<1, 32>>>();
    }) / n;
}

// B：4 条流轮转各发 250 次
static double bench_multi_stream_launch(int n = 1000) {
    const int NS = 4;
    cudaStream_t st[NS];
    for (int i = 0; i < NS; ++i) CK(cudaStreamCreate(&st[i]));

    double us = min_us([&] {
        for (int i = 0; i < n; ++i) {
            empty_kernel<<<1, 32, 0, st[i % NS]>>>();
        }
        for (int i = 0; i < NS; ++i) CK(cudaStreamSynchronize(st[i]));
    }) / n;

    for (int i = 0; i < NS; ++i) CK(cudaStreamDestroy(st[i]));
    return us;
}

// C：捕获 N 个空 kernel，测一次重放
static double bench_graph_replay(int n, double* out_batch_us) {
    cudaStream_t side;
    CK(cudaStreamCreate(&side));

    // 捕获前先真跑一次：让运行时把懒初始化做完，否则捕获期 malloc 会直接报错
    empty_kernel<<<1, 32, 0, side>>>();
    CK(cudaStreamSynchronize(side));

    cudaGraph_t graph;
    cudaGraphExec_t exec;
    CK(cudaStreamBeginCapture(side, cudaStreamCaptureModeThreadLocal));
    for (int i = 0; i < n; ++i) empty_kernel<<<1, 32, 0, side>>>();
    CK(cudaStreamEndCapture(side, &graph));
    CK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

    double batch = min_us([&] { CK(cudaGraphLaunch(exec, 0)); });
    CK(cudaDeviceSynchronize());

    if (out_batch_us) *out_batch_us = batch;
    double per_node = batch / n;

    CK(cudaGraphExecDestroy(exec));
    CK(cudaGraphDestroy(graph));
    CK(cudaStreamDestroy(side));
    return per_node;
}

// D：4 个 spin kernel，串行 vs 4 流并发
static void bench_spin_concurrency(long long cycles, double* serial_us, double* concurrent_us) {
    const int NS = 4;
    cudaStream_t st[NS];
    for (int i = 0; i < NS; ++i) CK(cudaStreamCreate(&st[i]));

    // 单发一次的成本（其中 1 个 kernel 的纯执行时间）
    double one = min_us([&] { spin_kernel<<<1, 32, 0, st[0]>>>((long long)cycles); });

    *serial_us = min_us([&] {
        for (int i = 0; i < NS; ++i) spin_kernel<<<1, 32, 0, st[0]>>>((long long)cycles);
    });

    *concurrent_us = min_us([&] {
        for (int i = 0; i < NS; ++i) spin_kernel<<<1, 32, 0, st[i]>>>((long long)cycles);
        for (int i = 0; i < NS; ++i) CK(cudaStreamSynchronize(st[i]));
    });

    for (int i = 0; i < NS; ++i) CK(cudaStreamDestroy(st[i]));

    printf("  （单个 spin kernel 实测 %.1f µs；%lld cycles）\n", one, cycles);
}


int main() {
    int dev = 0;
    CK(cudaSetDevice(dev));
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, dev));
    double clk_ghz = prop.clockRate / 1e6;   // kHz -> GHz
    printf("设备: %s\n", prop.name);
    printf("SM=%d  标称主频=%.2f GHz  显存=%.1f GB\n\n",
           prop.multiProcessorCount, clk_ghz,
           prop.totalGlobalMem / 1073741824.0);

    // ---------- A ----------
    double per_launch = bench_serial_launch(1000);
    printf("A. launch 成本（单流，1000 次连发）\n");
    printf("   每次提交 = %.2f µs\n\n", per_launch);

    // ---------- B ----------
    double per_launch_ms = bench_multi_stream_launch(1000);
    printf("B. 多流提交（4 stream，空 kernel）\n");
    printf("   每次 = %.2f µs   相对单流 %+.1f%%\n", per_launch_ms,
           (per_launch_ms - per_launch) / per_launch * 100.0);
    printf("   注：空 kernel 没有活可并行，这里的收益只可能来自 CPU 提交流水线\n\n");

    // ---------- C ----------
    printf("C. CUDA Graph 重放（图内 N 个空 kernel）\n");
    printf("   %4s %12s %12s %10s\n", "N", "整批 µs", "每节点 µs", "比 A 省 µs");
    printf("   %s\n", "--------------------------------------------");
    const int ns[] = {1, 2, 4, 8, 16, 32, 64};
    double sum_x = 0, sum_y = 0, sum_xx = 0, sum_xy = 0;
    int cnt = 0;
    for (int n : ns) {
        double batch = 0;
        double per = bench_graph_replay(n, &batch);
        printf("   %4d %12.1f %12.2f %10.2f\n", n, batch, per, per_launch - per);
        sum_x += n; sum_y += batch; sum_xx += (double)n * n; sum_xy += n * batch; ++cnt;
    }
    double den = cnt * sum_xx - sum_x * sum_x;
    double slope = (cnt * sum_xy - sum_x * sum_y) / den;      // µs / 节点
    double intercept = (sum_y - slope * sum_x) / cnt;          // µs
    printf("\n   拟合: 整批 µs ≈ %.1f（固定） + %.2f × N\n", intercept, slope);
    printf("   → 重放一张图固定 %.1f µs，图内每节点 %.2f µs\n", intercept, slope);
    printf("   → 按 C++ 口径，graph 每节点净收益 = %.2f − %.2f = %.2f µs\n",
           per_launch, slope, per_launch - slope);
    if (per_launch > slope) {
        printf("   → 拐点 N* = %.1f / %.2f ≈ %.1f（小于它用 graph 反而亏）\n",
               intercept, per_launch - slope, intercept / (per_launch - slope));
    }
    printf("\n");

    // ---------- D ----------
    // 让单个 spin kernel 跑约 1.5 ms：cycles ≈ 时长 × 实测频率
    long long cycles = (long long)(1.5e-3 * clk_ghz * 1e9);
    double serial = 0, concurrent = 0;
    printf("D. 真并发（4 个 spin kernel）\n");
    bench_spin_concurrency(cycles, &serial, &concurrent);
    printf("   串行 4 次 = %.1f µs\n", serial);
    printf("   4 流并发  = %.1f µs\n", concurrent);
    printf("   加速 = %.2fx（理想 4.00x，效率 %.0f%%）\n", serial / concurrent,
           serial / concurrent / 4.0 * 100.0);
    printf("   注：每个 spin kernel 只占 1 个 SM，%d 个 SM 里用 1 个 → 并发的空间来自空 SM\n\n",
           prop.multiProcessorCount);

    printf("读法：\n");
    printf("  1) A 是“每次提交多少钱”，C 的斜率是“图内每次多少钱”，两者之差就是 graph 的收益上限。\n");
    printf("  2) B 若明显快于 A，是提交流水线的功劳；D 快才是 GPU 真并行。\n");
    printf("  3) 把 A 的数字带进你的 decode：1040 µs / 次，A=%.2f µs ⇒ 占比 %.1f%%\n",
           per_launch, per_launch / 1040.0 * 100.0);

    CK(cudaDeviceReset());
    return 0;
}
