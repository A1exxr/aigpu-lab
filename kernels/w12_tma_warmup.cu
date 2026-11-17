/*
  tma_warmup.cu —— TMA 最小单测（D4，只在 Hopper 上能跑；在 4060 上只能编译）

  验证编译器能否为 Hopper 架构生成 TMA 指令（UTMALDG）和 mbarrier 指令（MBAR）

  这个文件的目的只有一个：让 SASS 里出现 UTMALDG
  4060 是 sm_89，没有 TMA 单元，所以这里不做运行验证，只做代码写对了、编译器能生成 TMA 指令 的静态验收：

    nvcc -arch=sm_90a -cubin -o /tmp/tma.cubin kernels/tma_warmup.cu \
         -I. -Xptxas -v
    cuobjdump -sass /tmp/tma.cubin | grep -c UTMALDG     # 期望 > 0
    cuobjdump -sass /tmp/tma.cubin | grep -cE 'MBAR'     # 期望 > 0（mbarrier）

  真正跑要等 D9 租 H100：`nvcc -arch=sm_90a -o tma tma_warmup.cu -lcuda`

  ── TMA 的三件套 ────────────────────────────
  1) host 侧：cuTensorMapEncodeTiled 生成一个 128 B 的 描述符，
    TMA 不是靠线程算地址，而是靠一个 128字节的opaque数据结构(叫 Tensor Map)来描述要搬什么、怎么搬
    这个描述符在 CPU 端用 cuTensorMapEncodeTiled API 生成
    这里面编码了：基址、各维长度、stride、元素类型、 每次搬多大的 tile、swizzle 模式
    描述符必须 64 字节对齐，且作为 kernel 参数时，必须用 __grid_constant__ 传

     
  2) device 侧：cp.async.bulk.tensor.2d 指令
    这是一条 PTX 指令，功能是 按照描述符的指示，把一个 2D tile 从 global memory 搬到 shared memory
    地址由 TMA 硬件自己算，整块搬运，不是逐元素拷贝，支持越界自动零填充
    可在搬运过程中直接输出 swizzle 布局
  

  3) 完成通知：mbarrier
    TMA 是完全异步的；发起指令后线程立刻返回，数据什么时候到齐由 mbarrier 通知
    流程：
      发 TMA 前：调 mbarrier.arrive.expect_tx(bytes) 登记“我即将搬 N 字节”
      TMA 硬件搬完数据后：自动把 N 字节从 mbarrier 的待传输计数中扣除
      Consumer 线程：调 mbarrier.try_wait.parity 等待计数归零
     

  ── 与 cp.async 的差别 ────────────
  cp.async：不经寄存器，但地址仍由线程算，且只能一维连续
  TMA    ：地址由硬件算，支持 2D/3D、越界自动零填充、直接输出 swizzle 布局
           wgmma 要的正是 swizzle 过的 smem，所以 TMA + wgmma 是配套的

*/
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

#define CUDA_CHECK(x)                                                          \
  do {                                                                         \
    cudaError_t e_ = (x);                                                      \
    if (e_ != cudaSuccess) {                                                   \
      printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
    }                                                                          \
  } while (0)

constexpr int TILE_H = 32;   // 一次搬 32 行
constexpr int TILE_W = 64;   // 每行 64 个 half（128 B，正好一个 swizzle 最小原子单位）

// ───────────────────────── 五个 PTX 工具 ─────────────────────────
// 这五个函数是对 Hopper 新 PTX 指令的 inline assembly 封装

__device__ __forceinline__ void mbar_init(uint64_t *bar, uint32_t count) {
  uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(a), "r"(count));
}

// 登记「这一轮我要收 bytes 个字节」；arrive 一次让计数 -1
__device__ __forceinline__ void mbar_arrive_expect_tx(uint64_t *bar, uint32_t bytes) {
  uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(a), "r"(bytes));
}

// parity 等待：phase 0/1 交替，第 n 次等 n%2
__device__ __forceinline__ void mbar_wait(uint64_t *bar, uint32_t phase) {
  uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "WAIT_%=:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      "@!p bra WAIT_%=;\n"
      "}\n" ::"r"(a), "r"(phase));
}

// 2D TMA 装载：把 tile 放到 smem 的 (x, y) 起点
__device__ __forceinline__ void tma_load_2d(const CUtensorMap *map, void *smem,
                                            int x, int y, uint64_t *bar) {
  uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  uint32_t b = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3}], [%4];" ::"r"(s),
      "l"(reinterpret_cast<uint64_t>(map)), "r"(x), "r"(y), "r"(b)
      : "memory");
}

// 一维 bulk 拷贝：不走描述符，直接给 global 地址与字节数（对照用）
__device__ __forceinline__ void bulk_load_1d(void *smem, const void *gmem,
                                             uint32_t bytes, uint64_t *bar) {
  uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  uint32_t b = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1], %2, [%3];" ::"r"(s),
      "l"(reinterpret_cast<uint64_t>(gmem)), "r"(bytes), "r"(b)
      : "memory");
}

// ───────────────────────── kernel ①：2D TMA ─────────────────────────

__global__ void tma_warmup_kernel(const __grid_constant__ CUtensorMap tmap,
                                 const __half *__restrict__ src,
                                 float *__restrict__ out, int H, int W) {
  // 目标 buffer 必须 128 B 对齐（TMA 的硬要求）
  __shared__ alignas(128) __half tile[TILE_H * TILE_W];
  __shared__ alignas(8) uint64_t bar;

  const int tid = threadIdx.x;
  if (tid == 0) mbar_init(&bar, 1);   // 只等 1 次 arrive（tid 0 自己）
  __syncthreads();                    // init 必须对所有线程可见

  const int x = blockIdx.x * TILE_W;  // 列起点（元素为单位）
  const int y = blockIdx.y * TILE_H;  // 行起点

  if (tid == 0) {
    // 先登记要收多少字节，再发指令（顺序不能反，否则 barrier 提前完成）
    mbar_arrive_expect_tx(&bar, TILE_H * TILE_W * sizeof(__half));
    tma_load_2d(&tmap, tile, x, y, &bar);
  }

  // 所有线程都等这一个 barrier；等到了说明 tile 已经在 smem 里
  mbar_wait(&bar, 0);

  // 用一个能吃的量证明数据到齐了：整块求和
  float s = 0.f;
  for (int i = tid; i < TILE_H * TILE_W; i += blockDim.x)
    s += __half2float(tile[i]);

  // 只让 block 0 写回，方便 host 侧和 CPU 对账
  if (blockIdx.x == 0 && blockIdx.y == 0 && tid == 0) out[0] = s;

  // 越界演示：tile 起点贴到 (W-TILE_W, H-TILE_H) 之外时，
  // TMA 会按 oobFill 填 0（这里用 NONE = 零填充），不会崩
  (void)src; (void)H; (void)W;
}

// ───────────────────────── kernel ②：1D bulk 对照 ─────────────────────────

__global__ void bulk_warmup_kernel(const __half *__restrict__ src,
                                   float *__restrict__ out) {
  __shared__ alignas(128) __half buf[TILE_H * TILE_W];
  __shared__ alignas(8) uint64_t bar;

  const int tid = threadIdx.x;
  if (tid == 0) mbar_init(&bar, 1);
  __syncthreads();

  if (tid == 0) {
    const uint32_t bytes = TILE_H * TILE_W * sizeof(__half);
    mbar_arrive_expect_tx(&bar, bytes);
    bulk_load_1d(buf, src, bytes, &bar);
  }
  mbar_wait(&bar, 0);

  float s = 0.f;
  for (int i = tid; i < TILE_H * TILE_W; i += blockDim.x)
    s += __half2float(buf[i]);
  if (tid == 0) out[1] = s;
}

// ───────────────────────── host 侧：建描述符 + 启动 ─────────────────────────

// 返回 0 表示成功。CUDA 12 的签名，注意 globalStrides 只有 rank-1 项、单位是字节。
static int make_tensor_map(CUtensorMap *tmap, void *ptr, int H, int W) {
  cuuint64_t gdim[2] = {(cuuint64_t)W, (cuuint64_t)H};        // {列, 行}
  cuuint64_t gstride[1] = {(cuuint64_t)W * sizeof(__half)};    // 第 1 维的步长（字节）
  cuuint32_t box[2] = {(cuuint32_t)TILE_W, (cuuint32_t)TILE_H};
  cuuint32_t estride[2] = {1, 1};

  CUresult r = cuTensorMapEncodeTiled(
      tmap, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, /*rank=*/2, ptr, gdim, gstride, box,
      estride, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) {
    const char *msg = "unknown";
    cuGetErrorString(r, &msg);
    printf("cuTensorMapEncodeTiled failed: %s\n", msg);
    return -1;
  }
  return 0;
}

// 这个函数在 4060 上不被调用（调用会 CUDA error 或直接崩），
// 留着是为了 D9 租到 H100 后一条命令就能跑。
extern "C" int tma_warmup_run(int H, int W) {
  __half *d_src = nullptr;
  float *d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_src, (size_t)H * W * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&d_out, 2 * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_src, 0, (size_t)H * W * sizeof(__half)));

  // 描述符必须 64 B 对齐
  alignas(64) CUtensorMap tmap;
  if (make_tensor_map(&tmap, d_src, H, W) != 0) return -1;

  dim3 grid((W + TILE_W - 1) / TILE_W, (H + TILE_H - 1) / TILE_H);
  tma_warmup_kernel<<<grid, 128>>>(tmap, d_src, d_out, H, W);
  bulk_warmup_kernel<<<1, 128>>>(d_src, d_out);
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaFree(d_src));
  CUDA_CHECK(cudaFree(d_out));
  return 0;
}
