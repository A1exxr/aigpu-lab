/*
  tma_warmup.cu —— TMA(张量内存加速器) 最小单测（D4，只在 Hopper 上能跑；在 4060 上只能编译）

  验证编译器能否为 Hopper 架构生成 TMA 指令（UTMALDG）和 mbarrier 指令（MBAR）

  这个文件的目的只有一个：让 SASS 里出现 UTMALDG
  4060 是 sm_89，没有 TMA 单元，所以这里不做运行验证，只做代码写对了、编译器能生成 TMA 指令 的静态验收：
  用 cuobjdump -sass 反汇编 cubin，grep 指令名确认生成正确。真正运行需要 H100

    TMA必须用90a，不是普通的90
    nvcc -arch=sm_90a w12_tma_warmup.cu -o w12_tma_warmup -lcuda    ———— 普通编译，用下面三个步骤

    在kernel目录下
    nvcc -arch=sm_90a -cubin -o /tmp/tma.cubin w12_tma_warmup.cu -I. -Xptxas -v             编译并导出硬件二进制（Cubin）  
    cuobjdump -sass /tmp/tma.cubin | grep -c UTMALDG            # 期望 > 0                  反汇编并验证 TMA 搬运指令       grep -c UTMALDG 结果为 1：TMA 搬运指令生成成功
    cuobjdump -sass /tmp/tma.cubin | grep -cE 'ARRIVES|BAR'     # 期望 > 0（mbarrier）      验证异步同步机制（mbarrier）    grep -cE 'ARRIVES|BAR' 结果为 2：2 个同步指令

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
#include <cuda.h>                 // CUDA Driver API（cuTensorMapEncodeTiled 在这里）
#include <cuda_fp16.h>
#include <cuda_runtime.h>         // CUDA Runtime API（cudaMalloc 等）
#include <cstdint>
#include <cstdio>

#define CUDA_CHECK(x)                                                          \
  do {                                                                         \
    cudaError_t e_ = (x);                                                      \
    if (e_ != cudaSuccess) {                                                   \
      printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
    }                                                                          \
  } while (0)

constexpr int TILE_H = 32;   // Height，行，一次搬 32 行
constexpr int TILE_W = 64;   // Width，列，每行 64 个 half（128 B，正好一个 swizzle 最小原子单位）

// ───────────────────────── 五个 PTX 工具 ─────────────────────────
// 这五个函数是对 Hopper 新 PTX 指令的 inline assembly 封装
// 每个都标记了 __device__ __forceinline__，确保内联到调用处，不会产生函数调用开销

/*
  static_cast 编译期类型转换的关键字
  static_cast<目标类型>(待转换的值或表达式)
  什么时候用：
    显式的数值类型转换：将索引 int 转换为 size_t，或者在混合精度计算时将 float 显式转为 half
    已知类型的 void* 还原：比如从 CUDA 的 cudaMalloc 或某些回调函数中拿到的 void*，明确知道它原本是什么类型时，用 static_cast<T*> 转回来

  
  reinterpret_cast 位模式重解释
  什么时候用：
    底层内存的字节级操作（Type Punning）：比如需要把一个 float 数组的底层比特位直接当成 uint32_t 数组来处理
    硬件寄存器/共享内存地址强转：在编写底层 CUDA 内核时，有时需要将共享内存的指针强制转换为特定的底层类型以进行向量化加载（如 float4），或者将地址转为整数进行对齐检查
*/

// 硬件异步屏障(mbarrier)初始化封装
__device__ __forceinline__ void mbar_init(uint64_t *bar, uint32_t count) {
  // 将 generic 指针转成 shared memory 的32位偏移地址
  // PTX 的 mbarrier 指令要求操作数是 shared memory 的32位地址
  // 在CUDA里 uint32_t 和 unsigned 是等价的 都是 32 位
  uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(bar));

  // asm volatile 告诉编译器这是一段内联汇编
  // mbarrier.init.shared::cta.b64：这是 Hopper (SM90) 及更新架构引入的硬件级异步屏障初始化指令
  // mbarrier：表示这是一个内存屏障对象，通常占用8字节
  // .init：初始化操作，设定该屏障需要等待多少个线程或异步操作到达
  // .shared::cta：表示这个 mbarrier 的作用域是当前 CTA（block）内的 shared memory
  // CTA（Cooperative Thread Array，协作线程阵列）就是 block 线程块
  
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(a), "r"(count));
  /*  
    初始化mbarrier对象，设置 pending_count = count，tx_count = 0，phase = 0
    这3个属性是Hopper架构下 mbarrier 对象的核心状态机
      1、pending_count = count  线程到达倒计时
        含义：表示当前阶段（Phase）还需要多少个线程（或硬件单元）调用 arrive 操作，才能宣告这一轮同步完成
          初始化时设为 count，表示这一轮同步预期有多少次 arrive
          每当有一个线程执行 mbarrier.arrive，这个值就原子减 1；当它减到 0 时，说明所有预期的线程都已经到达了同步点

      2、tx_count = 0  异步事务字节账本
        含义：用于追踪异步内存拷贝的字节数
          初始位为 0，当线程发起 TMA 异步拷贝并调用 arrive.expect_tx 时，会将预期传输的字节数加到 tx_count
          当底层的 TMA 硬件引擎完成实际的数据搬运后，会自动将这些已完成的字节数从 tx_count 中减去
          当 tx_count 归零时，说明所有预期的异步数据都已经安全落地到 Shared Memory 中

      3、phase = 0  同步周期的轮次标记
        含义：用于区分当前屏障处于第几轮同步周期（通常只有 0 和 1 两个状态，即奇偶相位）
          初始为 0；当 pending_count == 0 且 tx_count <= 0 同时满足时，当前阶段完成
          硬件会自动将 phase 翻转(0变1，1变0)，并将 pending_count 重置为初始的 count，tx_count 重置为 0，从而无缝进入下一轮同步
          消费者线程通过 try_wait 检查 phase 是否翻转，来判断自己是否可以安全地读取 Shared Memory 中的数据
    
    当前阶段完成 = 所有线程都到了 (pending_count == 0) 且 所有异步数据都搬完了 (tx_count <= 0)
    一旦满足这个条件，phase 自动翻转，唤醒等待的线程，同时计数器自动重置，为下一轮循环做好准备
  */
}


// 生产者线程(Producer)发起 TMA 异步拷贝时的核心同步指令 封装
__device__ __forceinline__ void mbar_arrive_expect_tx(uint64_t *bar, uint32_t bytes) {
  uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(bar));

  // mbarrier.arrive：当前线程向 mbarrier 报到，硬件会自动将 pending_count 减 1
  // .expect_tx：Hopper 架构的关键特性，告诉 mbarrier 线程到位且发起了一个异步任务，记一个账 ——> 硬件把传入的 bytes 加到 mbarrier 的 tx_count 上
  // _下划线：在PTX中，表示丢弃该指令的返回值
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(a), "r"(bytes));
  /*
    通常出现在 Producer Warp 中
    工作流：
      1、发起拷贝：线程调用 TMA 指令，告诉硬件引擎：把 Global Memory 的这块数据搬到 Shared Memory
      2、登记账本：线程立刻调用 mbar_arrive_expect_tx，告诉 mbarrier：我发起了搬运，预期有 bytes 字节的数据
      3、硬件接管：TMA 引擎在后台默默搬运。每写完一批数据，硬件会自动从 mbarrier 的 tx_count 中减去对应的字节数
      4、完成：当 pending_count == 0 且 tx_count <= 0 时，mbarrier 的当前 phase 自动翻转，唤醒等待的 Consumer 线程

    expect_tx 机制将 线程的执行进度和硬件的数据搬运进度 绑定在了同一个 mbarrier 对象上，这是实现计算与通信重叠（Overlap）且不发生数据竞争的基础
    这个指令得在 cp.async.bulk.tensor (TMA 指令) 之前调用，如果顺序反了，TMA 硬件可能在 tx_count 还是 0 时就尝试扣减，导致未定义行为或死锁
  */ 
}

// parity 等待：phase 0/1 交替，第 n 次等 (n-1)%2
// 消费者线程(Consumer)等待异步数据就绪的核心阻塞原语
// 这就是 Hopper/Blackwell 架构下实现 mbarrier 阻塞等待最标准、最底层的写法
__device__ __forceinline__ void mbar_wait(uint64_t *bar, uint32_t phase) {
  uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile(
    "{\n"
    ".reg .pred p;\n"
    "WAIT_%=:\n"
    // mbarrier.try_wait.parity：检查 mbarrier 的当前 phase 是否已经翻转到目标值。如果未完成，挂起当前 warp
    "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
    "@!p bra WAIT_%=;\n"
    "}\n" ::"r"(a), "r"(phase));
}

/*
  核心思想：
    时序：expect_tx 必须在 TMA 之前
    状态：pending_count 管线程，tx_count 管字节
    复用：用 parity (奇偶相位) 来区分多轮流水线
*/


// 2D TMA 装载：把 tile 放到 smem 的 (x, y) 起点
// TMA（张量内存加速器）发起 2D 异步数据搬运的核心 PTX 指令封装
__device__ __forceinline__ void tma_load_2d(const CUtensorMap *map, void *smem,
                                            int x, int y, uint64_t *bar) {
  uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  uint32_t b = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile(
    /*
      cp.async.bulk.tensor.2d：表示发起一次 2D 张量的批量异步拷贝
      .shared::cluster.global：明确了数据流向，从全局内存到共享内存，cluster 表示支持线程块集群内的共享内存访问
      .mbarrier::complete_tx::bytes：搬运完成后，自动去更新指定的 mbarrier 对象的 tx_count
      [%0]：目标地址，即共享内存中接收数据的起始地址（smem）
      [%1, {%2, %3}]：源地址描述  
        %1 是主机端预先构建好的 CUtensorMap（张量描述符）的 64 位地址
        %2 和 %3 是 2D 张量中的 Tile 坐标（x, y）
      [%4]：同步对象地址，即之前初始化的 mbarrier 在共享内存中的地址（bar）
    */
    "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
    " [%0], [%1, {%2, %3}], [%4];" ::"r"(s),
    "l"(reinterpret_cast<uint64_t>(map)), "r"(x), "r"(y), "r"(b)
    : "memory");
}
/*
  这段代码通常由 Producer Warp 中的单个线程执行
  工作流：
    1、前置记账：先调用 mbar_arrive_expect_tx，告诉 mbarrier 预期要搬运多少字节
    2、发起搬运：执行这段 tma_load_2d，TMA 硬件引擎接管，开始从 Global Memory 搬运数据到 Shared Memory
    3、自动扣款：TMA 搬运完成后，硬件自动触发 mbarrier::complete_tx，将 tx_count 归零
    4、唤醒消费者：tx_count 归零且 pending_count 归零后，mbarrier 的 phase 翻转，Consumer 线程通过 mbar_wait 被唤醒，开始执行 MMA 计算
*/


// 一维 bulk 拷贝：一般不走描述符，直接给 global 地址与字节数
__device__ __forceinline__ void bulk_load_1d(void *smem, const void *gmem,
                                             uint32_t bytes, uint64_t *bar) {
  uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  uint32_t b = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile(
    // 指令去掉了.tensor后缀，意味着不需要在主机端预先构建CUtensorMap描述符，直接按字节流进行物理地址的线性拷贝
    // 4个占位符：目标地址、源地址、搬运的字节数、同步对象地址
    "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
    " [%0], [%1], %2, [%3];" ::"r"(s),
    "l"(reinterpret_cast<uint64_t>(gmem)), "r"(bytes), "r"(b)
    : "memory");
}

// ───────────────────────── kernel 1：2D TMA ─────────────────────────

// __grid_constant__ 参数限定符：被限定的参数存储在 常量内存 中，对整个 grid 的所有线程只读可见
// 每个 grid launch 有独立的副本，不同 grid 之间互不干扰
// grid 的数量由调用 Kernel 的次数决定； launch多个kernel就有多个grid
__global__ void tma_warmup_kernel(const __grid_constant__ CUtensorMap tmap,
                                 const __half* __restrict__ src,
                                 float* __restrict__ out, int H, int W) {
  // TMA 要求 smem目标地址 128 B 对齐
  // tile大小 32*64*2B = 4096B = 4KB
  __shared__ alignas(128) __half tile[TILE_H * TILE_W];   // 目标地址
  // mbarrier是64位对象，必须8字节对齐
  __shared__ alignas(8) uint64_t bar;

  const int tid = threadIdx.x;

  // 只有 tid==0 时初始化 mbarrier；count=1表示只等 1 次 arrive(tid 0自己)
  if (tid == 0) 
  {
    mbar_init(&bar, 1); 
  }
  __syncthreads();                    // init 必须对所有线程可见

  // 计算当前 block 负责的 tile 的全局起始坐标
  // TILE_H 搬 32 行； TILE_W 每行 64 个 half
  // TMA 的坐标系是“物理内存坐标系”，而不是“数学矩阵坐标系”
  // TMA 的 x 轴：代表内存地址连续变化的方向
  // TMA 的 y 轴：代表内存地址跳跃（跨行）的方向
  const int x = blockIdx.x * TILE_W;  // 列起点（元素为单位）
  const int y = blockIdx.y * TILE_H;  // 行起点

  // 发起 TMA 装载，只有 tid == 0 执行
  if (tid == 0) 
  {
    // 先登记要收多少字节，再发指令（顺序不能反，否则 barrier 提前完成）
    mbar_arrive_expect_tx(&bar, TILE_H * TILE_W * sizeof(__half));
    tma_load_2d(&tmap, tile, x, y, &bar);
  }

  // 所有线程都等这一个 barrier
  // 当 mbarrier 的 pending_count == 0 && tx_count == 0 时，phase 从 0 翻转到 1，wait 解除
  // 此时 tile 中的数据保证完整可用
  mbar_wait(&bar, 0);

  // 验证数据：所有线程协作对 tile 求和
  float s = 0.f;
  for (int i = tid; i < TILE_H * TILE_W; i += blockDim.x)
  {
    s += __half2float(tile[i]);
  }
    

  // 只让 block 0 写回，方便 host 侧和 CPU 对账
  if (blockIdx.x == 0 && blockIdx.y == 0 && tid == 0) out[0] = s;

  // 越界演示：tile 起点贴到 (W-TILE_W, H-TILE_H) 之外时，
  // TMA 会按 oobFill 填 0（这里用 NONE = 零填充），不会崩
  // 这里的(void)是一个强制类型转换，把变量 src 转换成了 void 类型(无类型)
  // 就是 声明这个变量暂时不用，让编译器别报错
  (void)src; (void)H; (void)W;
}

// ───────────────────────── kernel 2：1D bulk 对照 ─────────────────────────

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

// 构建 TMA 描述符
static int make_tensor_map(CUtensorMap *tmap, void *ptr, int H, int W) {
  /*
    cuuint64_t 是 cuda定义的一个 无符号64位整数类型 就是 uint64_t
    gdim  {W, H}  全局张量维度；X-major：第0维是最内层(列/W)，第一维是外层 
    gstride 只有 rank-1 个元素（第 0 维隐式连续），单位是 字节
    globalStrides 只有 1 个元素，因为它是 2D 矩阵，硬件默认最内层是连续的，只需要告诉它跨行跳跃的字节数就行
    
    box	{TILE_W, TILE_H}	每次搬运的 tile 尺寸，同样 X-major

    estride	{1, 1}	元素步长，通常为 1（密集排列）
  */
  
  cuuint64_t gdim[2] = {(cuuint64_t)W, (cuuint64_t)H};            // {列, 行}
  cuuint64_t gstride[1] = {(cuuint64_t)W * sizeof(__half)};       // 第 1 维的步长（字节）
  cuuint32_t box[2] = {(cuuint32_t)TILE_W, (cuuint32_t)TILE_H};
  cuuint32_t estride[2] = {1, 1};

  // 把上面这些信息打包生成最终的描述符
  // CUresult 是 CUDA Driver API 所有函数统一使用的返回值类型，它是一个枚举，用来表示函数调用的成功或失败状态
  CUresult r = cuTensorMapEncodeTiled(
    tmap, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2,                 // 基础信息：输出的描述符对象、数据类型、张量的维度(2D矩阵就是2)
    ptr, gdim, gstride,                                       // 全局内存布局组：全局内存的起始地址ptr、全局张量总尺寸、外层维度的跨步长
    box, estride,                                             // 每次把搬运的tile尺寸、元素步长
    CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,    // 性能优化与边界组
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
// 留着是为了 D9 租到 H100 后一条命令就能跑
// extern "C" 是一个链接规范，告诉 C++ 编译器：请按照 C 语言的规则来编译和链接这个函数，不要使用 C++ 的规则
extern "C" int tma_warmup_run(int H, int W) {
  __half *d_src = nullptr;
  float *d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_src, (size_t)H * W * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&d_out, 2 * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_src, 0, (size_t)H * W * sizeof(__half)));

  // 描述符必须 64 B 对齐
  alignas(64) CUtensorMap tmap;
  // r == 0 就是 CUDA_SUCCESS
  if (make_tensor_map(&tmap, d_src, H, W) != 0) return -1;

  dim3 grid((W + TILE_W - 1) / TILE_W, (H + TILE_H - 1) / TILE_H);
  tma_warmup_kernel<<<grid, 128>>>(tmap, d_src, d_out, H, W);
  bulk_warmup_kernel<<<1, 128>>>(d_src, d_out);
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaFree(d_src));
  CUDA_CHECK(cudaFree(d_out));
  return 0;
}

int main()
{
  const int H = 128;
  const int W = 128;

  tma_warmup_run(W, H);


  return 0;
}