"""台阶 4 入口：先验证 fp16 的正确性，再测速。"""
from w11_attn_bwd import bench_step4, test_step4

if __name__ == "__main__":
    test_step4()
    bench_step4()
