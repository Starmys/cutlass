import os

import torch
import triton


M = 5120
N = 5120
K = 4096


def profile(fn, args, warmup=25, rep=100):
    return triton.testing.do_bench(lambda: fn(*args), warmup=warmup, rep=rep)


if __name__ == '__main__':
    x = torch.rand((M, K), dtype=torch.float16, device='cuda')
    w = torch.rand((N, K), dtype=torch.float16, device='cuda')
    print(f'({profile(torch.nn.functional.linear, [x, w])})ms')
