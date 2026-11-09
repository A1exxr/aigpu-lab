# SASS 怎么读——只看三列就够

> 写于 2026-11-09。所有"实测"数字都来自本机（RTX 4060 Laptop / sm_89 / CUDA 12.8）你自己的编译产物。

---

## 1. SASS 是什么

你写的是 CUDA C++，显卡看不懂。显卡真正执行的是一串**机器指令**，这串指令就叫 **SASS**（Streaming ASSembler）。

```
你的 .cu ──nvcc──► PTX（虚拟指令，跨代兼容） ──ptxas──► SASS（硬件真指令） ──► 显卡执行
```

| | CPU | GPU |
|---|---|---|
| 你写的 | C++ | CUDA C++ |
| 中间层 | 汇编 | PTX |
| **硬件真正执行** | **x86 机器码** | **SASS** |

三句话记住：

1. **SASS 是显卡的指令集**，一套指令集 + 你程序编译出来的那串指令。
2. **你不用写它**，nvcc 全自动生成。它是产物。
3. 看它只为一个目的：**验证编译器有没有按你的意图生成指令**（你以为它一次搬 16 字节，它可能偷偷搬了 4 次 2 字节）。

命令行示例（你 `k_gelu.so` 里真实存在的两行）：

```
LDG.E.U16  R2, [R4.64]      ← 从显存读 2 个字节到寄存器 R2
STG.E.U16  [R6.64], R2      ← 把 R2 里那 2 个字节写回显存
```

---

## 2. SASS 和 ncu 不是一回事

| | **SASS** | **ncu** |
|---|---|---|
| 本质 | 编译产物（静态） | 测量工具（动态） |
| 要跑 kernel 吗 | **不要**，只读文件，连显卡都不用 | **要**，必须真跑一遍 |
| 回答什么 | "代码被编译成了什么指令" | "这次运行花了多久、卡在哪" |
| 类比 | 看菜谱：写了几刀几火 | 站厨房掐表：炒了几分钟 |
| 看一次够吗 | 每次重新编译都要重看 | 每次跑都要重测 |

**配合方式**：ncu 说"卡在等全局内存"（`long_scoreboard` 高），SASS 说"因为 4 条装载都是 2 字节，一条 warp 指令只搬 64 B"。**ncu 定位，SASS 解释。**

---

## 3. 怎么拿到 SASS

先修 PATH（这台机器上 `cuobjdump`/`ncu` 都不在 PATH）：

```bash
export PATH=/usr/local/cuda/bin:$PATH
```

### 3.1 用脚本（推荐，只看计数）

```bash
# 在仓库根目录（WSL 里是 ~/projects/aigpu-lab）
export PATH=/usr/local/cuda/bin:$PATH
PY=~/miniconda3/envs/triton/bin/python

# 自动找 ~/.cache/torch_extensions 下最新的 .so
$PY tools/_sass.py

# 指定文件
$PY tools/_sass.py ~/.cache/torch_extensions/py311_cu128/k_gelu/k_gelu.so

# 一次跑六个工程
bash tools/_sass_multi.sh
```

工具本体：[`tools/_sass.py`](../../tools/_sass.py)（约 100 行）、[`tools/_sass_multi.sh`](../../tools/_sass_multi.sh)（批量）。

### 3.2 直接命令

```bash
# 你的 kernel 都在这里（load_inline 的产物）
ls ~/.cache/torch_extensions/py311_cu128/

# dump 整个 .so 的 SASS
cuobjdump -sass ~/.cache/torch_extensions/py311_cu128/k_gelu/k_gelu.so > /tmp/gelu.sass

# 先列出所有函数名（mangled）
grep "Function :" /tmp/gelu.sass

# 只看一个函数
cuobjdump -sass -fun _Z11gelu_kernelPK6__halfPS_i x.so

# 寄存器数 / spill（比看 SASS 更快的第一步）
cuobjdump -res-usage x.so | grep -A5 "Function :"

# 单独编译的 cubin（nvdisasm 输出更干净）
nvcc -arch=sm_89 -cubin -o k.cubin k.cu && nvdisasm -c k.cubin
```

### 3.3 Triton kernel（w11 / paged 那些）

```python
# 编译后把 cubin 落盘，再用 nvdisasm
open("/tmp/k.cubin", "wb").write(compiled.asm["cubin"])
```

### 3.4 带源码对照

`ncu` 图形界面的 **Source** 页：左边 C++、右边 SASS、每行带采样计数。想"这行慢在哪"就用它。

---

## 4. 输出表逐列解释

> 列名是脚本起的，**不是 SASS 的标准**。每一格统计的是"这种指令出现了几条"，单位是**条**不是字节。

| 表头 | SASS 长什么样 | 中文 | 干什么 | 见到多少算正常 |
|---|---|---|---|---|
| **LDG总** | `LDG.E` | 全局读 | 从显存读 | 逐元素几条；分块后很少 |
| **LDG128** | `LDG.E.128` | 16 字节读 | 一条搬 16 B | >0 且其余为 0 = 向量化到位 |
| **LDG64** | `LDG.E.64` | 8 字节读 | 一条搬 8 B | 看数据类型 |
| **LDGU16** | `LDG.E.U16` | 2 字节读 | 一条只搬 2 B | 逐元素 half kernel 会有；多了说明没打包 |
| **STG总** | `STG.E` | 全局写 | 往显存写 | 与 LDG 对称 |
| **STGU16** | `STG.E.U16` | 2 字节写 | 一条写 2 B | 同上 |
| **LDL** | `LDL` | local 读 | 从 local memory 读 | **应为 0**；非 0 = 踩坑 |
| **STL** | `STL` | local 写 | 往 local memory 写 | **应为 0** |
| **LDS** | `LDS` | 共享内存读 | 从 `__shared__` 读 | 用了分块就该有，几十条正常 |
| **STS** | `STS` | 共享内存写 | 往 `__shared__` 写 | 同上 |
| **HMMA** | `HMMA.16816` | 张量核心矩阵乘 | 一条算一小块矩阵乘加 | 用 wmma/mma 就 >0 |
| **LDSM** | `LDSM` | 矩阵装载 | 把共享内存按 fragment 布局喂给 HMMA | 跟着 HMMA 出现 |
| **FFMA** | `FFMA` | 乘加 | 一次 `a*b+c`，CUDA core 的活 | 逐元素几十条；纯 TC 的应为 0 |
| **BAR** | `BAR.SYNC` | 屏障 | `__syncthreads()` 的落地 | 约等于代码里**有效**屏障数 |
| **LDGSTS** | `LDGSTS` | cp.async | 显存→共享内存，不过寄存器 | 用了 `cp.async` 就 >0 |
| **SHFL** | `SHFL` | warp 内交换 | `__shfl_down_sync` 等 | 归约里每轮 1 条 |
| **判定P** | `@P0 LDG...` | 谓词指令 | 带 `if` 条件执行 | 少 = `if` 提到循环外了 |

### 4.1 「总」与细分是包含关系

```
LDG总 = 所有 LDG.E 的条数
LDG128 / LDG64 / LDGU16 = 其中宽度为 16 B / 8 B / 2 B 的那些
```

`LDG总=4, LDG128=0, LDG64=0, LDGU16=0` 读作：**4 条装载，全是 32 位（4 B）**。

### 4.2 换算「每线程搬多少字节」

```
每线程读出字节
  = LDG128×16 + LDG64×8 + (LDG总 − LDG128 − LDG64 − LDGU16)×4 + LDGU16×2
```

再乘 32（一个 warp 的 lane 数）= **一条 warp 指令的访存足迹**。

---

## 5. 三个真实输出怎么读

### 5.1 gelu 三个版本（`k_gelu.so`）

```
kernel                          LDG总  LDG128  LDG64  LDGU16  STG总  STGU16  LDL  STL  LDS  STS  HMMA  LDSM  FFMA  BAR  LDGSTS  SHFL  判定P
_Z11gelu_kernelPK6__halfPS_i        4       0      0       0      4       0    0    0    0    0     0     0    56    0       0     0      0
gelu_naive_kernelPK6__halfPS_i      1       0      0       1      1       1    0    0    0    0     0     0     7    0       0     0      0
gelu_kernel_float4PK6__halfPS_i     1       1      0       0      1       0    0    0    0    0     0     0    56    0       0     0      0
```

| kernel | 每线程读 | 解释 |
|---|---|---|
| `gelu_naive_kernel` | `1×2` = **2 B** | 1 个 half；warp 一条指令 64 B |
| `gelu_kernel`（`__half2`） | `4×4` = **16 B** | 8 个 half，但用了 **4 条 32 位**；warp 一条指令 128 B |
| `gelu_kernel_float4` | `1×16` = **16 B** | 1 条 128 位，指令数最少；但写回只有 4 B（不对称） |

**交叉验证技巧**：`FFMA ÷ 7 = 每线程处理的元素数`。naive `7÷7=1`、另两版 `56÷7=8`，全都对得上。

**结论**：`__half2` 版（81.4%）的收益不是"向量化到 128 位"（它 `LDG128=0`），而是**把每次 2 B 换成 4 B**，让一条 warp 指令从 64 B 变成 128 B —— 这是**合并访存**的功劳。

### 5.2 GEMM 四个台阶（`k_gemm.so`）

```
kernel                    LDG总 LDGU16  STG  LDS  STS  HMMA  LDSM  FFMA  BAR
gemm_naive_kernel           10    10     1    0    0     0     0     5    0
gemm_tiled_kernel            2     2     1   40    2     0     0    32    2
gemm_reg_kernel             14    14    16    6   14     0     0    32    2
gemm_tc_kernel              14    14     5    9   18     4     4     0    3
```

- **naive**：`LDS/STS=0` → 完全没分块
- **tiled**：`LDS=40, STS=2` → 一次搬进 smem，之后反复读 smem
- **reg**：`HMMA=0, FFMA=32` → CUDA core 上做 4×4
- **tc**：`HMMA=4, LDSM=4, FFMA=0` → 矩阵乘全交给张量核心

**判读口诀**：`LDS/STS` 看有没有分块；`HMMA` 看有没有用张量核；**`FFMA` 归零**才说明计算真从 CUDA core 转到 TC 上。

### 5.3 屏障数可以直接对账

```
k_softmax:  BAR=4, SHFL=21, LDS=7, STS=6
k_reduce :  BAR=5, SHFL=5,  LDS=9, STS=5
```

`softmax` 的 `BAR.SYNC=4` 正好等于优化后的"4 次屏障"；`reduce` 是 5。**报告里的数字能被 SASS 数出来**，这就是"我看过"的证据。

### 5.4 张量核心长什么样（`wmma_demo.so`）

```
_Z12mma_one_tilePK6__halfS1_Pf     LDG总=8  HMMA=2  LDSM=0  FFMA=0
```

`HMMA=2` 就是"矩阵乘真的走张量核心"的落地指令。

---

## 6. 判读速查表

| SASS 里看到 | 含义 | 下一步 |
|---|---|---|
| `LDG.E.U16` 很多 | 一元素一条指令，2 B/次 | 换 `__half2` / `uint4` |
| `LDG.E.128` 出现 | 每线程 16 B | 好；还慢就查事务是否跨行 |
| `LDL` / `STL` 出现 | local memory | `-Xptxas -v` 看 spill，或检查 `&局部变量` |
| `LDS/STS` 多、`BAR` 少 | 分块充分、同步不贵 | 好 |
| `LDS/STS` 少、`LDG` 多 | 没分块 | 上 tiling |
| `HMMA > 0` 但 `FFMA` 也多 | 边界/填充在吃 CUDA core | 调 tile 让 K 整除 |
| `@P0` 遍布循环体 | 边界判断没提到循环外 | 主循环去 `if`，尾块单独处理 |
| `LDGSTS` | `cp.async` 生效 | 好；再看流水深度 |

---

## 7. 三个坑

**1. SASS 只对"某一次编译"有效。**
同一份源码换了 nvcc 版本、`-O` 级别、甚至只是 ptxas 心情不同，指令都可能变。`k_gelu.so` 里 `gelu_kernel_float4` 的 `LDL=0` 就说明**这次编译**没把 `box` 物化——你当年测到的 19.3% 慢，要么是另一份产物，要么慢因另有其他。**结论必须与当次产物对应。**

**2. `U16` 多 ≠ 慢。**
`U16` 只在"warp 内地址散开"时浪费带宽：
- 相邻 lane 地址连续（`x[i]`）→ 一条指令 64 B 连续，扇区**全用满**，只是多花指令；
- 相邻 lane 隔 128 B（跨行读 K）→ 一条指令取 32 个扇区 1024 B，只用 64 B → **6.25%**。

**同一个 `LDG.E.U16`，两种命运。判断看地址表达式，不看指令本身。**

**3. 函数名是 mangled 的。**
`_Z11gelu_kernelPK6__halfPS_i` = `gelu_kernel(const __half*, __half*, int)`。先 `grep "Function :"` 列全，再用 `-fun` 精确 dump。

---

## 8. 五分钟练习

```bash
export PATH=/usr/local/cuda/bin:$PATH
PY=~/miniconda3/envs/triton/bin/python
# 在仓库根目录执行（WSL: ~/projects/aigpu-lab）

# ① 三个 gelu 版本的装载宽度（对比 2 B / 4 B / 16 B）
$PY tools/_sass.py ~/.cache/torch_extensions/py311_cu128/k_gelu/k_gelu.so

# ② 张量核心长什么样
$PY tools/_sass.py ~/.cache/torch_extensions/py311_cu128/wmma_demo/wmma_demo.so

# ③ 97.6% 的 paged attn 每 lane 只读 2 B，却仍贴带宽墙
$PY tools/_sass.py ~/.cache/torch_extensions/py311_cu128/k_paged_attn_split/k_paged_attn_split.so
```

**看完 ①②③ 就够用了，不必再往下读指令内容。** SASS 是解释工具，不是新项目。

---

## 9. 面试口径

> "CUDA C++ 到硬件之间编译器会改你的意图，所以我会 dump SASS 验证。比如 gelu 的逐元素版是 `LDG.E.U16`（每线程 2 B），`__half2` 版是 4 条 32 位装载（每线程 16 B），一条 warp 指令的足迹从 64 B 变成 128 B——这就是带宽利用率从 63% 到 81% 的机制。再配合 `LDL/STL` 判断有没有掉进 local memory，`-Xptxas -v` 看寄存器数和 spill。"
