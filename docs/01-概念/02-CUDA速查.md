# CUDA 复习速查：写 RMSNorm 只需要的那些

> **用途**：感觉自己"全忘了"的时候，用这份做 90 分钟定向复习。
> **纪律**：**不要打开任何 CUDA 教程书。** 你需要的只有下面 6 条 + 一个归约骨架。
> **判据**：第 5 节的 6 道自测能全答对，就立刻回去写 kernel。答不上来的那一条，只看那一条。

---

## 0. 先接受一件事

你"忘干净"的感觉，大部分来自**没有具体任务时回忆知识**。
一旦有具体任务（写一行 RMSNorm），需要的知识其实只有下面这一页。

**CUDA 的知识面很大，但写这个 kernel 用到的部分很小。**

---

## 1. 六条清单（写完就能停）

| # | 要点 | 复习到什么程度算够 |
|---|---|---|
| 1 | **线程层次与索引** | 能默写 `i = blockIdx.x * blockDim.x + threadIdx.x`，能解释 `gridDim.x` 是什么 |
| 2 | **kernel 声明与 launch 语法** | 能默写 `__global__ void k(...)` + `k<<<blocks, threads>>>(...)`，知道 `__restrict__` 只是给编译器优化的提示 |
| 3 | **合并访存（coalescing）** | 能说清为什么"相邻线程访问相邻地址"快，"跨步访问"慢 |
| 4 | **shared memory** | 能写 `__shared__ float s[256];`，知道它是**每个 block 一份**，块内线程共享 |
| 5 | **`__syncthreads()`** | 知道它是块内屏障，知道**绝对不能放在会分叉的 `if` 里** |
| 6 | **block 内树形归约** | 能默写下面第 3 节的骨架 —— **这是唯一的"要背"的东西** |

**这六条之外的一切（warp shuffle、向量化、occupancy 计算、bank conflict 细节）今天都不用复习。**

---

## 2. 前五条：一页讲完

### ① 线程层次与索引

```
grid（网格）
 └── 由若干 block（线程块）组成          ← blockIdx.x 是"我是第几个 block"
      └── 每个 block 由若干 thread 组成  ← threadIdx.x 是"我在块内排第几"
```

```cpp
// 一维铺开（逐元素 kernel 的写法）
int i = blockIdx.x * blockDim.x + threadIdx.x;

// 按行划分（RMSNorm 要的写法）
int row = blockIdx.x;        // 一个 block 负责一行
int col = threadIdx.x;       // 块内线程负责行内的列
```

`blockDim.x` = 一个 block 里有多少线程（就是你 launch 时写的 `threads`）。
`gridDim.x` = 一共有多少个 block。

**grid 要向上取整**，否则尾部元素没人处理：
```cpp
int blocks = (n + threads - 1) / threads;   // 整数除法向上取整
```

### ② kernel 声明与 launch

```cpp
__global__ void my_kernel(const float* __restrict__ in,
                          float* __restrict__ out,
                          int n) {
    // 这里跑在 GPU 上，每个线程都执行一遍
}

// 宿主端启动：三个尖括号 = 执行配置 <<<grid, block>>>
my_kernel<<<blocks, threads>>>(in_ptr, out_ptr, n);
```

- `__global__`：GPU 上执行、CPU（宿主）调用
- `__restrict__`：告诉编译器"这个指针不会和别的指针指向同一块内存"，让它敢做优化。**写上是习惯，不加不错，加了更好。**

### ③ 合并访存（coalescing）

GPU 从显存取数据是以 **cache line（128 字节）** 为单位的。一个 warp（32 个线程）如果访问**连续的** 32 个 float（=128 字节），硬件**一次事务**就取回来了。

```cpp
out[i] = x[i] * 2;      // ✅ 相邻线程访问相邻地址 → 1 次事务
out[i] = x[i * 2] * 2;  // ❌ 跨步访问 → 要 32 个不同的 cache line
```

**这条是"为什么你的带宽利用率可能很低"的最常见原因。**

### ④ shared memory

```cpp
__shared__ float s[256];   // 每个 block 一份，块内所有线程都能读写
```

- 生命期 = 这个 block 的生命期
- **比 global memory 快一到两个数量级**（在芯片上，不在显存里）
- 用途：**块内线程之间交换数据**——比如把 256 个线程各自算出的部分和放进来，再做归约

```cpp
int tid = threadIdx.x;
s[tid] = 局部计算结果;      // 每个线程写自己那一格
```

### ⑤ `__syncthreads()`

```cpp
__syncthreads();   // 屏障：block 内所有线程都到这里，才继续往下走
```

两个必须记住的规则：

1. **写完 shared memory 之后、读别人写的那一格之前，必须有 `__syncthreads()`。** 否则你读到的是没写完的数据 → **结果随机错，重跑不一样**。
2. **绝对不能放在会分叉的 `if` 里。** 如果一部分线程进了 `if`、一部分没进，进不去的那部分永远到不了屏障，进去的那部分永远在等 → **死锁 / 未定义行为**。

```cpp
// ❌ 错误
if (tid < stride) {
    s[tid] += s[tid + stride];
    __syncthreads();          // 下一轮 stride 变小，一半线程进不来 → 死锁
}

// ✅ 正确
if (tid < stride) {
    s[tid] += s[tid + stride];
}
__syncthreads();              // 放在 if 外面，所有线程都执行
```

---

## 3. 第六条：块内树形归约骨架（唯一要背的）

这不是 RMSNorm 的答案，**这是一个通用的"块内求和"**。看懂它，然后自己想怎么套进 RMSNorm。

```cpp
// 把数组 x 按 block 分段求和，每个 block 输出一个和
__global__ void block_sum(const float* __restrict__ x,
                          float* __restrict__ out,
                          const int n) {
    __shared__ float s[256];              // 假设 blockDim.x == 256

    const int tid = threadIdx.x;
    const int i   = blockIdx.x * blockDim.x + tid;

    // ── 第一步：每个线程把自己那份装进共享内存
    s[tid] = (i < n) ? x[i] : 0.0f;       // 越界填 0，不能跳过不写！
    __syncthreads();

    // ── 第二步：树形归约，每轮活跃线程减半
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s[tid] += s[tid + stride];
        }
        __syncthreads();                  // ← 在 if 外面
    }

    // ── 第三步：线程 0 把结果写出去
    if (tid == 0) {
        out[blockIdx.x] = s[0];
    }
}
```

**三个容易写错的地方：**

| 位置 | 易错点 | 后果 |
|---|---|---|
| `s[tid] = (i < n) ? x[i] : 0.0f;` | 越界时 `if` 掉不写 | 共享内存里是未初始化的垃圾 → 随机错 |
| `__syncthreads()` 的位置 | 放进 `if` 里 | 死锁 / 随机错 |
| 循环 `stride` 初值 | 写成 `blockDim.x` 而不是 `/2` | 第一轮自己和自己相加，结果翻倍 |

**树形归约的直觉**：256 个部分和 → 128 个 → 64 → ... → 1。总共 log₂(256) = 8 轮。

---

## 4. 你还需要知道的一个"模式"：广播

归约完之后，结果只在 `s[0]` 里。**但你需要 block 里所有线程都拿到它**（因为每个线程都要用它算自己负责的元素的输出）。

```cpp
if (tid == 0) {
    s[0] = 最终结果;
}
__syncthreads();          // ← 必须！否则别的线程读到旧值
float val = s[0];         // 现在所有线程都拿到同一个值了
```

**这个"归约 → 广播"两步，是几乎所有归一化 kernel（LayerNorm / RMSNorm / Softmax）的公共骨架。**

---

## 5. 自测：6 题，答得出就不用再复习了

> 先自己答，再翻到第 6 节对答案。**答不上来的那一条，就是你今天唯一要补的。**

1. 一个 block 有 256 个线程，grid 有 2048 个 block。`blockIdx.x == 2`、`threadIdx.x == 99` 的线程，全局索引 `i` 是多少？

2. `__syncthreads()` 写在 `if (tid < stride) { ... }` **里面**会怎样？为什么？

3. `__shared__ float s[256];`，线程 `tid` 读 `s[tid]` 有 bank conflict 吗？读 `s[tid * 2]` 呢？

4. `out[i] = x[i] * 2` 和 `out[i] = x[i * 2] * 2`，哪个快？为什么？

5. RMSNorm 里，一个 block 算完一行的平方和后，怎么让块内 256 个线程都拿到 `rms` 这个值？

6. 树形归约的循环里，如果 `__syncthreads()` 漏写了（只写 `s[tid] += s[tid+stride]`），会出现什么现象？

---

## 6. 自测答案

<details>
<summary>点开对答案（先自己答完）</summary>

**1.** `i = 2 * 256 + 99 = 611`
公式：`i = blockIdx.x * blockDim.x + threadIdx.x`。

**2.** **死锁 / 未定义行为。**
因为 `stride` 每轮减半，后面几轮只有前几十个线程满足 `tid < stride`。没进 `if` 的线程永远到不了那个 `__syncthreads()`，而进了 `if` 的线程会在屏障上一直等它们。**屏障必须在所有线程都执行到的位置。**

**3.** `s[tid]` **没有**冲突——相邻地址落在相邻 bank。
`s[tid * 2]` **有 2-way bank conflict**——32 个线程只用到 16 个 bank，每个 bank 要服务 2 个线程，访问被串行化。
（共享内存有 32 个 bank，每个 4 字节宽。知道这个概念就够，今天不用深究。）

**4.** **`out[i] = x[i] * 2` 快。**
一个 warp 的 32 个线程访问连续的 32 个 float = 128 字节 = 恰好一个 cache line，硬件**一次事务**取回。
`x[i*2]` 是跨步访问，32 个线程落在 32 个不同的 cache line 上，要 32 次事务。**有效带宽会掉到 1/32 量级。**

**5.** 把结果写进 shared memory，同步，然后所有线程一起读：
```cpp
if (tid == 0) s[0] = rms;
__syncthreads();
float my_rms = s[0];      // 所有线程都拿到
```
**注意 `__syncthreads()` 不能省**——省了的话别的线程可能在 `s[0]` 被写之前就读了。

**6.** **结果随机错，重跑结果不一样。**
因为没有屏障，快的线程已经在读 `s[tid + stride]`，而那个位置可能还没被对应的线程写完。这是典型的**竞态条件（race condition）**。诊断特征：**时对时错、每次结果都不同**。

</details>

---

## 7. 复习到什么程度就该停

**全部答对 → 立刻回去写 `w2_rmsnorm.cu`。**

如果你发现某一题不会：
- 只补那一题对应的那一节（第 2 节 ①–⑤ 或第 3 节）
- **不要顺手把整篇 CUDA 教程读一遍** —— 那是最常见的拖延

**时间上限：90 分钟。** 到了还没复习完也要开始写；写的时候遇到不会的再回来查，**带着问题查比空着脑子背快十倍**。

---

*配套文档：`docs/01-概念/01-如何写第一个kernel.md`（增量阶梯 + 按症状查坑）、`kernels/README.md`（5 个必须回答的问题）。*
