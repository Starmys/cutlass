import time
import torch

K = 4096
N = 11008

dtype=torch.float16

torch.manual_seed(2023)

a = torch.rand((N, K), dtype=dtype, device='cuda')
b = torch.rand((K, 32), dtype=dtype, device='cuda')

c = torch.mm(a, b)

torch.cuda.synchronize()
for _ in range(20):
    c = torch.mm(a, b)
torch.cuda.synchronize()
# start = time.time()
# for _ in range(100):
#     c = torch.mm(a, b)
# torch.cuda.synchronize()
# end = time.time()

# print(f'{(end - start):.6f} ms')
