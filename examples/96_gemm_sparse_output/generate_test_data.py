import os

import torch
import triton


EXAMPLE_PATH = os.path.join('examples', '96_gemm_sparse_output')
TARGET_PATH = os.path.join(
    os.path.dirname(__file__).replace(EXAMPLE_PATH, os.path.join('build', EXAMPLE_PATH)),
    'data'
)
os.makedirs(TARGET_PATH, exist_ok=True)


M = 2024
N = 5120
K = 4096
L = 5120
# M = 512
# N = 1536
# K = 1536
# L = 4096


def save_tensor(tensor: torch.Tensor, path: str, format: str):
    array = tensor.flatten().cpu().numpy()
    with open(path, 'w') as f:
        f.write(' '.join(map(lambda x: format % x, array)))


def profile(fn, args, warmup=5, rep=10):
    return triton.testing.do_bench(lambda: fn(*args), warmup=warmup, rep=rep)


if __name__ == '__main__':
    x = torch.randn((M, K), dtype=torch.float16, device='cuda')
    w = torch.randn((N, K), dtype=torch.float16, device='cuda')
    b = torch.randn((N, ), dtype=torch.float16, device='cuda')
    i = torch.randperm(L, dtype=torch.int64, device='cuda')[:M]
    y = torch.zeros((L, N), dtype=torch.float16, device='cuda')

    o = torch.nn.functional.linear(x, w, b)
    y.scatter_(0, i[:, None].expand(M, N), o)

    save_tensor(x.to(torch.float32), os.path.join(TARGET_PATH, 'A.txt'), '%.6f')
    save_tensor(w.to(torch.float32), os.path.join(TARGET_PATH, 'B.txt'), '%.6f')
    save_tensor(y.to(torch.float32), os.path.join(TARGET_PATH, 'C.txt'), '%.6f')
    save_tensor(b.to(torch.float32), os.path.join(TARGET_PATH, 'D.txt'), '%.6f')
    save_tensor(i.to(torch.int32), os.path.join(TARGET_PATH, 'I.txt'), '%d')

    print(f'Linear : ({profile(torch.nn.functional.linear, [x, w, b])})ms')
    print(f'Scatter: ({profile(y.scatter_, [0, i[:, None].expand(M, N), o])})ms')

    cmd = os.path.join('.', EXAMPLE_PATH, '96_gemm_sparse_output')
    cmd += f' {M} {N} {K} {L} {TARGET_PATH}'
    with open(os.path.join(TARGET_PATH, 'cmd.txt'), 'w') as f:
        f.write(cmd)
