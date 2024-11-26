import os

import torch
import triton
from flash_attn.flash_attn_interface import flash_attn_func


EXAMPLE_PATH = os.path.join('examples', '95_gemm_topk')
TARGET_PATH = os.path.join(
    os.path.dirname(__file__).replace(EXAMPLE_PATH, os.path.join('build', EXAMPLE_PATH)),
    'data'
)
os.makedirs(TARGET_PATH, exist_ok=True)


# M = 128
# N = 512
# K = 128
# L = 512
M = 65536
N = 65536
K = 128
L = 512


def save_tensor(tensor: torch.Tensor, path: str, format: str):
    array = tensor.flatten().cpu().numpy()
    with open(path, 'w') as f:
        f.write(' '.join(map(lambda x: format % x, array)))


def profile(fn, args, warmup=25, rep=100):
    return triton.testing.do_bench(lambda: fn(*args), warmup=warmup, rep=rep)


if __name__ == '__main__':
    q = torch.randn((M, K), dtype=torch.float16, device='cuda')
    k = torch.randn((N, K), dtype=torch.float16, device='cuda')
    v = torch.randn((N, K), dtype=torch.float16, device='cuda')
    # q = torch.arange(M, dtype=torch.float16, device='cuda').unsqueeze(1).tile((1, K))
    # k = torch.ones((N, K), dtype=torch.float16, device='cuda') / K

    p = torch.nn.functional.linear(q, k)
    # y = torch.topk(p, )

    save_tensor(q.to(torch.float32), os.path.join(TARGET_PATH, 'A.txt'), '%.6f')
    save_tensor(k.to(torch.float32), os.path.join(TARGET_PATH, 'B.txt'), '%.6f')
    # save_tensor(y.to(torch.float32), os.path.join(TARGET_PATH, 'C.txt'), '%.6f')

    print(f'Torch-Linear : ({profile(torch.nn.functional.linear, [q, k])})ms')
    q = q.reshape((1, M, 1, K))
    k = k.reshape((1, M, 1, K))
    v = v.reshape((1, M, 1, K))
    print(f'Flash-Attn   : ({profile(flash_attn_func, [q, k, v])})ms')

    cmd = os.path.join('.', EXAMPLE_PATH, '95_gemm_topk')
    cmd += f' {M} {N} {K} {L} {TARGET_PATH}'
    with open(os.path.join(TARGET_PATH, 'cmd.txt'), 'w') as f:
        f.write(cmd)
