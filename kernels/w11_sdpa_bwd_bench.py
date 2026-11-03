import torch, torch.nn.functional as F, triton.testing

B, H, S, D = 1, 8, 1024, 64
q, k, v = (torch.randn(B, H, S, D, device="cuda", dtype=torch.float16).requires_grad_(True) for _ in range(3))
do = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)

o = F.scaled_dot_product_attention(q, k, v)
t = min(triton.testing.do_bench(lambda: torch.autograd.grad(o, (q, k, v), do,  retain_graph=True)) for _ in range(3))
print(f"SDPA 反向 = {t:.3f} ms")
print(f"我的手写 = 0.514 ms  →  {t/0.514:.2f}× 慢于 / 快于 PyTorch")