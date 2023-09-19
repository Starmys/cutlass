import time
import torch

K = 4096
N = 4096

dtype=torch.float32

torch.manual_seed(2023)

a = torch.rand((N, K), dtype=dtype, device='cuda')
b = torch.rand((K, 1), dtype=dtype, device='cuda')

c = torch.mm(a, b)

torch.cuda.synchronize()
for _ in range(200):
    c = torch.mm(a, b)
torch.cuda.synchronize()
start = time.time()
for _ in range(1000):
    c = torch.mm(a, b)
torch.cuda.synchronize()
end = time.time()

print(f'{(end - start):.6f} ms')
