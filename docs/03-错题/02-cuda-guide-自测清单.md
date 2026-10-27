# CUDA 读书笔记 · 自测清单

> 配套文件：`docs/05-附录/cuda-guide-读书笔记.txt`（你 9 月读 CUDA C++ Programming Guide 第 2–3 章的笔记，UTF-8 转码版，1197 行）
> 生成日期：2026-10-25

## 怎么用（三条规则）

1. **只看你标了「重要 / 务必理解 / 必须理解」的章节**，其余一律跳过（见文末「跳过清单」）。
2. 每题给自己 **30 秒**口述。说不出来 → 在题号前打 ❌，当天记进 `docs/03-错题/01-CUDA错题本.md`。
3. **先答后看答案**。答案在最后，别倒着读。

一共 **25 题**，全部做完约 15 分钟。

---

## A. 内存与访存（5 题）

**A1**（笔记 L21 / L30）GPU 的存储层次从快到慢排序？共享内存的作用域和生命周期分别是什么？

**A2**（L59）什么叫「合并访存」？一个 warp 的 32 个线程分别访问 `a[tid]`（float，连续）和 `a[tid*32]`（跨 32 个 float），各产生多少次 32B 扇区请求？

**A3**（L72）矩阵转置：为什么「直接读 A 写 B」的版本慢？用 tile 之后慢在哪一步被消掉了？

**A4**（L92）共享内存 bank conflict 的成因是什么（几个 bank、每个多宽）？为什么 `float sm[32][32]` 有冲突、`float sm[32][33]` 没有？**如果是 `half sm[32][32]`，改成 `[32][33]` 还有效吗？**

**A5**（L784）L1 和共享内存是什么关系？怎么调整两者的分配比例？

## B. 执行与占用率（3 题）

**B1**（L116）occupancy 由哪三个硬件资源决定？`__launch_bounds__` 能影响其中哪个？

**B2**（L635）硬件多线程：为什么「SM 上驻留更多 warp」能提高带宽利用率？停滞（stall）时 SM 做什么？

**B3**（L3）`__shared__`、`__constant__`、`__device__`、`__restrict__` 各自是什么？哪些是存储空间，哪些只是修饰符？

## C. 流、事件与计时（4 题）

**C1**（L165）为什么需要多个 stream？两个 stream 里的 kernel 在什么条件下才能真的同时执行？

**C2**（L233）写出用 event 计时的四行骨架。为什么它比「`cudaDeviceSynchronize()` + CPU 计时」更准？

**C3**（L241）`cudaEventSynchronize` 和 `cudaStreamWaitEvent` 的区别？各自的使用场景？

**C4**（L195）`cudaStreamSynchronize` 和 `cudaDeviceSynchronize` 的区别？

## D. 默认流与同步（4 题）

**D1**（L311）legacy default stream（stream 0）有哪两条特殊语义？为什么它会让「多流并发」失效？

**D2**（L323）per-thread default stream 是什么？怎么开启？它和 legacy 的关键差别？

**D3**（L333）显式同步有哪几种粒度（线程块内 / 流内 / 设备级）？分别对应哪个 API 或内建？

**D4**（L544）隐式同步：举两个「看起来没同步、实际会同步」的典型场景。

## E. 固定内存与统一内存（4 题）

**E1**（L428）page-locked（pinned）内存为什么能让 `cudaMemcpyAsync` 真正异步？pageable 内存的 `cudaMemcpy` 为什么不行？

**E2**（L445）mapped memory（零拷贝）是什么？什么场景下它比「拷到显存再算」更快？

**E3**（L413）unified memory 的页迁移机制是什么？为什么第一次访问会慢？支持超出显存容量（oversubscription）吗？

**E4**（L495）pinned / mapped / unified 三者分别该在什么场景选？

## F. 多卡（3 题）

**F1**（L907）P2P 传输（`cudaMemcpyPeer`）和 P2P 内存访问（kernel 直接解引用对端指针）的区别？

**F2**（L942）做 P2P 内存访问前必须调用什么？一致性靠什么保证？

**F3**（L878）多卡编程的两种进程模型是什么？各自的典型用途？

## G. 环境变量与 SIMT（2 题）

**G1**（L573）`CUDA_LAUNCH_BLOCKING=1` 干什么用？什么场景下开它？

**G2**（L614）independent thread scheduling（Volta 起）解决了什么问题？为什么它让 `__syncwarp()` 变成必需？

---

## 答案要点

**A1** 寄存器 → 共享内存/L1（同一块物理 SRAM，可配置比例）→ L2 → 全局显存（DRAM）。共享内存作用域 = **一个 block**，生命周期 = **block 执行期间**（跟着 block 生灭）。
> 面试追问：线程私有的是什么？→ 寄存器 + local memory（local 物理在 DRAM，是"溢出/取地址"才用的兜底）。

**A2** 合并访存 = 一个 warp 的 32 个线程访问**连续**地址，硬件把它们合成尽量少的 32B 扇区请求。
- `a[tid]`：32×4B = 128B 连续 → **4 个 32B 扇区**（1 次 128B 事务）。
- `a[tid*32]`：每个线程跨 128B → 32 个不同扇区 → **32 次请求，每次只用 4B**（利用率 12.5%）。

**A3** 直接版：读 A 是行连续（好），写 B 是**列方向**——同一 warp 写 `B[col][row]` 时地址间隔 `N*4` 字节 → 完全非合并，每线程一次事务。tile 版先把 A 的 tile 读进共享内存（读仍是合并的），再让线程从共享内存里**转置着读**（共享内存没有"合并"概念，只有 bank conflict），写 B 就变成行连续了。**慢在"写 B 的非合并"被消掉了。**

**A4** 32 个 bank、每个 4 字节宽；同一 warp 内两个线程若访问**同一 bank 的不同地址**就串行化（conflict）。
- `[32][32]`：列方向 stride 32 → 32×4B=128B → 32 个地址落在**同一个 bank** → 32 路冲突。
- `[32][33]`：stride 33 → 33 mod 32 = 1 → 相邻行偏移一个 bank → 无冲突。
- **`half` 无效**：bank 宽 4B，`half` 只有 2B，两个 half 共享一个 bank；`[32][33]` 的 stride 是 66B = 16.5 个 bank → 未对齐，冲突照旧。fp16 要用 `[32][40]`（80B stride，20 bank，偶数对齐）或干脆按 `half2` 组织。**这条我们 10/23 实测过（K-06）。**

**A5** L1 和共享内存是**同一块物理 SRAM**，由 `cudaFuncSetAttribute(..., cudaFuncAttributePreferredSharedMemoryCarveout, ...)` 或 `cudaFuncAttributeMaxDynamicSharedMemorySize` 调整划分比例。共享内存要得多，L1 就小。

**B1** 三因素：① 每线程寄存器数 ② 每 block 共享内存 ③ 每 block 线程数（还有"每 SM 最大 block 数"这个上限）。
`__launch_bounds__(maxThreads, minBlocksPerSM)` 告诉编译器"每个 SM 至少塞下几个 block"，**编译器据此压低寄存器用量**（代价是可能溢出到 local）。

**B2** warp 因访存/依赖停滞时，SM 立刻切到另一个就绪 warp 发射指令（warp 级上下文切换是零开销的，因为寄存器是物理分区的）。所以在飞的请求越多、越能填满显存流水 → 带宽利用率越高。这也是为什么"提高 ILP（每线程多几个独立请求）"和"提高 occupancy"是两条不同的路。

**B3**
- `__shared__`：**存储空间**修饰，block 内共享。
- `__constant__`：**存储空间**，常量内存（64KB，广播友好）。
- `__device__`：**存储空间**，全局显存里的变量（所有线程共享一份）。
- `__restrict__`：**只是修饰符**（类似 C 的 restrict），承诺指针不重叠，让编译器敢做更激进的优化/向量化。

**C1** 单个 stream 内任务是串行的；多 stream 用来表达"这些任务互不依赖"。真正并发需要：**资源允许**（kernel 占用 SM 不满）+ **没有隐式同步**（不碰 legacy default stream、不用 pageable 拷贝）。

**C2**
```cpp
cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
cudaEventRecord(s, 0);
kernel<<<g,b>>>(...);
cudaEventRecord(e, 0);
cudaEventSynchronize(e);
float ms; cudaEventElapsedTime(&ms, s, e);
```
event 记录在 **GPU 时间线上**，量的是 kernel 在设备上的真实耗时；CPU 计时还包含启动开销和排队等待。

**C3** `cudaEventSynchronize(e)` = **主机**等这个 event 完成；`cudaStreamWaitEvent(stream, e)` = **让某个流**等这个 event（主机不等）。前者用于计时/取结果，后者用于跨流依赖。

**C4** `cudaStreamSynchronize(s)` 只等**这个流**；`cudaDeviceSynchronize()` 等**设备上所有流**。后者最省事也最容易把并发吃光。

**D1** legacy default stream（就是 `0`）的两条语义：① 它里面的操作**与所有"阻塞流"隐式串行**；② 它自己是阻塞流。所以只要有一个 kernel 丢进默认流，其他流的任务就被迫排队 → 多流并发失效。**这正是你 10/24 用 CUDA Graph capture 时遇到 "The CUDA Graph is empty" 的根因**（capture 期间不允许有落到默认流上的工作）。

**D2** per-thread default stream = 每个 host 线程有自己的"默认流"，且**不再和其他流隐式串行**。开启方式：编译加 `--default-stream per-thread`，或源文件里 `#define CUDA_API_PER_THREAD_DEFAULT_STREAM`。

**D3** 三种粒度：
- 线程块内：`__syncthreads()`（barrier）
- 流内：`cudaStreamSynchronize` / `cudaEventSynchronize`
- 设备级：`cudaDeviceSynchronize()`
（更细的还有 `__syncwarp()`、`__threadfence()`）

**D4** 典型隐式同步：
1. **pageable 内存**的 `cudaMemcpy`（同步版）——它会先等所有流，且拷贝本身是阻塞的；
2. `cudaMalloc` / `cudaFree`（会与设备同步）；
3. 默认流上的任何操作（见 D1）；
4. `cudaHostAlloc` 某些 flag 下的注册；
5. `cudaMemcpy` 与 kernel 在同一 stream 上天然有序（这不算隐式同步，是顺序保证）。

**E1** pinned 内存的物理页**不会被换出**，DMA 引擎可以直接访问，所以 `cudaMemcpyAsync` 能在内核执行的同时真异步传输。pageable 内存必须先由驱动拷进一个内部 pinned staging buffer，这一步无法异步 → `cudaMemcpyAsync` 退化成同步。

**E2** mapped memory = 把 host 的 pinned 内存**映射到设备地址空间**，kernel 直接读写（零拷贝，走 PCIe）。适合：数据只读一次、数据量小、或想省一次显式拷贝延迟的场景。8 GB 显存 + PCIe 带宽（~12 GB/s）远低于显存带宽时，大块数据仍然应该先拷进显存。

**E3** unified memory 是**按页迁移**的：`cudaMallocManaged` 先分配在 host，kernel 启动/首次访问时触发 page fault，把页搬到 GPU；`cudaDeviceSynchronize` 后再迁回。第一次访问慢正是因为 **page fault + 迁移**。**不支持 oversubscription**（不能分配超过显存的 managed 内存）。

**E4** 选型：需要**异步 DMA**（大块传输、多流重叠）→ pinned；**小数据、只读一次、要省一次拷贝** → mapped；**代码想省心、数据在 CPU/GPU 间频繁切换但量不大** → unified（代价是 page fault 抖动）。生产代码里，推理引擎几乎只用 **pinned + 显式异步拷贝**。

**F1** P2P **传输**：`cudaMemcpyPeer(dst, dstDev, src, srcDev, size)`，显式把数据从一张卡搬到另一张卡（走 NVLink/PCIe，不经过 host 内存）。P2P **访问**：kernel 里直接解引用**另一张卡**的指针（需要 `cudaDeviceEnablePeerAccess`），数据不动，算力跨卡访问远端显存。

**F2** 必须 `cudaDeviceEnablePeerAccess(peerDev, 0)`（双向都要）。一致性：P2P 访问是**弱一致**的，跨卡写之后要 `cudaDeviceSynchronize()`（或 event/stream 同步）才能被对端看到；P2P 传输本身在 stream 内有序。

**F3** 两种模型：
1. **单进程多设备**：一个进程 `cudaSetDevice` 切换，线程/流分别绑卡（简单，适合 2–4 卡工作站）；
2. **多进程 + MPI/NCCL**：每卡一个进程，通信走 NCCL（生产训练的标准做法）。

**G1** `CUDA_LAUNCH_BLOCKING=1` 让**所有 kernel 启动变成同步**（启动即等待完成）。用途：**定位是哪个 kernel 报的错**——异步执行时错误会延后到某次同步才爆出来，栈已经对不上了。

**G2** Volta 之前，一个 warp 的 32 个线程**锁步**执行（同一 PC）；Volta 起每个线程有独立 PC 和调用栈，warp 内可以发散后各自推进。于是"warp 内所有线程都到了某一行"不再是自动成立的，必须显式 `__syncwarp()`（或依赖 `__shfl_sync` 的 mask）。

---

## 跳过清单（你标了「暂时不看 / 了解 / 可忽略」）

atomics 细节、cooperative groups、`cudaLaunchKernelEx`、launching clusters、programmatic dependent launch、batched memory transfers、PTX、异步屏障、pipeline、CUDA driver API 的 kernel execution、多设备 managed memory。

**理由**：这些在 8 GB 单卡 + 校招面试的场景里出现概率极低；真要用（比如 clusters）等你上 H100 再说。
