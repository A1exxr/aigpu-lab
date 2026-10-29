#include <torch/extension.h>
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;         // WMMA 的命名空间




__global__ void gemm_nt_kernel(){}