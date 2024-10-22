import os

import torch
import triton


EXAMPLE_PATH = os.path.join('examples', '96_gemm_sparse_output')
TARGET_PATH = os.path.join(
    os.path.dirname(__file__).replace(EXAMPLE_PATH, os.path.join('build', EXAMPLE_PATH)),
    'data'
)
os.makedirs(TARGET_PATH, exist_ok=True)


M = 5120
N = 5120
K = 4096
# M = 128
# N = 256
# K = 64


def save_tensor(tensor: torch.Tensor, path: str, format: str):
    array = tensor.flatten().cpu().numpy()
    with open(path, 'w') as f:
        f.write(' '.join(map(lambda x: format % x, array)))


def profile(fn, args, warmup=25, rep=100):
    return triton.testing.do_bench(lambda: fn(*args), warmup=warmup, rep=rep)


if __name__ == '__main__':
    x = torch.rand((M, K), dtype=torch.float16, device='cuda')
    w = torch.rand((N, K), dtype=torch.float16, device='cuda')
    b = torch.rand((N, ), dtype=torch.float16, device='cuda')
    # x = torch.zeros((M, K), dtype=torch.float16, device='cuda')
    # w = torch.zeros((N, K), dtype=torch.float16, device='cuda')
    # x[0, 0:32] = 1
    # w[0, 0:32] = 1
    # x[0, 32:64] = 2
    # w[0, 32:64] = 2
    o = torch.nn.functional.linear(x, w, b)
    print(f'({profile(torch.nn.functional.linear, [x, w, b])})ms')

    save_tensor(x.to(torch.float32), os.path.join(TARGET_PATH, 'A.txt'), '%.6f')
    save_tensor(w.to(torch.float32), os.path.join(TARGET_PATH, 'B.txt'), '%.6f')
    save_tensor(o.to(torch.float32), os.path.join(TARGET_PATH, 'C.txt'), '%.6f')
    save_tensor(b.to(torch.float32), os.path.join(TARGET_PATH, 'D.txt'), '%.6f')

    cmd = os.path.join('.', EXAMPLE_PATH, '96_gemm_sparse_output')
    cmd += f' {M} {N} {K} {TARGET_PATH}'
    with open(os.path.join(TARGET_PATH, 'cmd.txt'), 'w') as f:
        f.write(cmd)
