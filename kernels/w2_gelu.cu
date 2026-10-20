#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

/**
 *     实现gelu，输入输出都是fp16，[1,2048,4096]
 *  没有归约、没有共享内存、没有屏障、没有跨线程通信
    block = 1024，thread = 512，一个线程处理8个元素
    但线程不通过for循环，而是通过float4(float4不是数据类型，是一次性搬16字节的箱子)
    fp16，线程一次性搬16个字节，就是8个元素，并且还是合并访问
    float4是cuda内置的 向量类型，包含4个float，共16字节 ————> 加载一次float4，相当于读了4个float
    这样是向量化访存，提高带宽利用率

    gelu里面的误差函数，cuda通过erff()实现，fp16 进来先 __half2float，算完 __float2half 写回

    float4 是 128 位向量化， 向量化：一个线程，用一条指令，搬运多于一个元素。
    __half2 是 32位
    half 是 16 位
*/


// 这是个错误例子，因为取了两个地址
// float4 慢不是因为向量化，而是因为 箱子里面装的是 float，而你的数据是 half
// 要"把 float 看成 half"→ 必须重解释内存 → 必须给地址 → local memory。
// 0.6390 ms    52.5 GB/s   19.3%   0.24x
__global__ void gelu_kernel_float4(const half* __restrict__ x,
                            half* __restrict__ out,
                            const int n4){      // n4是箱子个数，元素数/8，每个元素负责一个箱子
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    // 越界检查
    if(i >= n4){return;}
    
    // reintrepret_cast不改变数据，只是“换个类型解读同一块内存”
    // 把 x 当作 float4* 数组；[i] 取第 i 个箱子，即 16 字节，就是 8 个 half。
    float4 box = reinterpret_cast<const float4*>(x)[i];      
    // 把 box 的地址当作 half* 来解读；h[0..7] 就是那 8 个 half；没有数据拷贝，只是换个视角看同一块内存。
    half* h = reinterpret_cast<half*>(&box);     // 这样写，编译器就必须给 box 一个内存地址，非常慢
    // 局部变量默认住在【寄存器】里  →  而寄存器没有内存地址 —> 写了一次 &box  →  编译器必须给 box 一块【内存】= local memory
    // local memory 物理上在 DRAM 里（走 L1/L2 缓存）
    // 下面还有一个&obox，整个float4里面，local访存有 18 次 × 2 B
    
    // 声明一个输出箱子 obox
    float4 obox;
    half* oh = reinterpret_cast<half*>(&obox);

    // 遍历8个half，
    #pragma unroll
    for(int k = 0; k < 8; k++)
    {
        float temp = __half2float(h[k]);
        // 下面计算gelu
        // oh[k] = temp * 0.5 * (1 + erff( temp / sqrt(2)));                // 不能直接用sqrt(2)，sqrt 是双精度 double类型
        oh[k] = __float2half(temp * 0.5 * (1 + erff( temp * rsqrtf(2))));   // 快速倒数平方根
    }                                
    reinterpret_cast<float4*>(out)[i] = obox;
}


// 0.1956 ms   171.6 GB/s   63.1%   0.77x
__global__ void gelu_naive_kernel(const half* __restrict__ x,
                                  half* __restrict__ out,
                                  const int n)          // n = 元素数，不是箱子数
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = __half2float(x[i]);
    out[i] = __float2half(0.5f * v * (1.0f + erff(v * 0.70710678f)));
}




__global__ void gelu_kernel(const half* __restrict__ x,
                            half* __restrict__ out,
                            const int n4)          // n4 = 元素数 / 8（还是每线程 16 字节）
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n4) return;

    // 一"箱" = 16 字节 = 4 个 __half2
    // 用__half2(4 字节 = 2 个half) 作为向量单位，每个线程处理4个__half2 = 16字节 = 8个元素
    // x是const half* ————> 把 x 当作 __half2 数组的首地址，每个__half2就是两个half
    // 每个线程负责 4 个 __half2，所以指针偏移 (size_t)i * 4，跳过前 i 个线程负责的 4i 个 __half2；
    const __half2* x2 = reinterpret_cast<const __half2*>(x) + (size_t)i * 4;
    __half2*       o2 = reinterpret_cast<__half2*>(out) + (size_t)i * 4;

    const float INV_SQRT2 = 0.70710678f;      // 1/√2，提到循环外

    #pragma unroll      // 让编译器展开这 4 次循环，因为下面k=0,k<4，4次
    // 4 次循环就有 4 次判断 + 4 次自增 + 4 次跳转。展开后这些全没了
    for (int k = 0; k < 4; ++k)
    {
        float2 f = __half22float2(x2[k]);     // 4 字节读，一次拿 2 个 half，所以x2[k]返回2个half

        float2 g;
        // float2是 CUDA 内置的向量类型，定义为包含 x 和 y 两个float类型的成员的结构体
        g.x = 0.5f * f.x * (1.0f + erff(f.x * INV_SQRT2));
        g.y = 0.5f * f.y * (1.0f + erff(f.y * INV_SQRT2));

        o2[k] = __float22half2_rn(g);         // 4 字节写
    }
}





torch::Tensor gelu(torch::Tensor x){
    TORCH_CHECK(x.is_cuda(),"x必须在GPU上");
    TORCH_CHECK(x.scalar_type() == torch::kFloat16,"只支持 fp16");
    auto xc = x.contiguous();
    auto out = torch::empty_like(xc);
    const int n = xc.numel();
    const int n4 = n / 8;

    int threads = 512;
    int blocks = n4 / threads;

    // naive
    // int blocks = (n + threads - 1) / threads;


    gelu_kernel<<<blocks,threads>>>(    
        reinterpret_cast<const half*>(xc.data_ptr<at::Half>()),  
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),       
        n4);    

    // gelu_naive_kernel<<<blocks,threads>>>(    
    //     reinterpret_cast<const half*>(xc.data_ptr<at::Half>()),  
    //     reinterpret_cast<half*>(out.data_ptr<at::Half>()),       
    //     n);                                       
    return out;
}