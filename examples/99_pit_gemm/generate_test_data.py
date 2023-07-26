import os

import torch


EXAMPLE_PATH = os.path.join('examples', '99_pit_gemm')
TARGET_PATH = os.path.join(
    os.path.dirname(__file__).replace(EXAMPLE_PATH, os.path.join('build', EXAMPLE_PATH)),
    'data'
)

SHAPE_M = 256
SHAPE_K = 1024
SHAPE_N = 512
SPARSE_PORT = 'A'
TRANSPOSE_A = False
TRANSPOSE_B = True
BLOCK_H = 1
BLOCK_W = 8
SPARSITY = 0.9


def build_index(mask: torch.Tensor, block: tuple[int, int]):
    H, W = mask.shape
    BH, BW = block
    NH, NW = H // BH, W // BW
    mask = mask.reshape((NH, BH, NW, BW)).any(dim=1).any(dim=-1)
    index = []
    arange = torch.arange(mask.shape[-1], dtype=torch.int32, device=mask.device)
    for row in mask:
        cols = arange[row] + 1
        cols = cols[torch.randperm(cols.shape[0])]
        pad = torch.zeros(mask.shape[-1] - cols.shape[0], dtype=torch.int32, device=mask.device)
        cols = torch.concat([cols, pad])
        index.append(cols.unsqueeze(0))
    return torch.concat(index)


def generate_sdd_data(shape: tuple[int, int, int], sparsity: float):
    M, K, N = shape
    mask = torch.rand(size=(M, K), requires_grad=False, device='cuda') > sparsity
    A = torch.ones(size=(M, K), dtype=torch.float16, requires_grad=False, device='cuda') * mask
    B = torch.ones(size=(K, N), dtype=torch.float16, requires_grad=False, device='cuda')
    C = torch.matmul(A, B)
    return A, B, C, mask


def generate_dsd_data(shape: tuple[int, int, int], sparsity: float):
    M, K, N = shape
    mask = torch.rand(size=(K, N), requires_grad=False, device='cuda') > sparsity
    A = torch.randn(size=(M, K), requires_grad=False, device='cuda')
    B = torch.randn(size=(K, N), requires_grad=False, device='cuda') * mask
    C = torch.matmul(A, B)
    return A, B, C, mask


def generate_data(
    sparse_port: str,
    shape: tuple[int, int, int],
    block: tuple[int, int],
    sparsity: float,
    trans_A: bool,
    trans_B: bool,
):
    if sparse_port == 'A':
        A, B, C, mask = generate_sdd_data(shape, sparsity)
    elif sparse_port =='B':
        A, B, C, mask = generate_dsd_data(shape, sparsity)
    else:
        raise NotImplementedError(f'Unsupported sparse port: {sparse_port}')
    if trans_A:
        A = A.T.contiguous()
        if sparse_port == 'A':
            mask = mask.T.contiguous()
    elif trans_B:
        B = B.T.contiguous()
        if sparse_port == 'B':
            mask = mask.T.contiguous()
    idx = build_index(mask, block)
    return A, B, C, idx


def save_tensor(tensor: torch.Tensor, path: str, format: str):
    array = tensor.flatten().cpu().numpy()
    with open(path, 'w') as f:
        f.write(' '.join(map(lambda x: format % x, array)))


def save_data(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    idx: torch.Tensor,
    path: str,
):
    save_tensor(A.to(torch.float32), os.path.join(path, 'A.txt'), '%.6f')
    save_tensor(B.to(torch.float32), os.path.join(path, 'B.txt'), '%.6f')
    save_tensor(C.to(torch.float32), os.path.join(path, 'C.txt'), '%.6f')
    save_tensor(idx, os.path.join(path, 'idx.txt'), '%d')


if __name__ == '__main__':
    torch.manual_seed(2023)
    A, B, C, idx = generate_data(
        sparse_port=SPARSE_PORT,
        shape=[SHAPE_M, SHAPE_K, SHAPE_N],
        block=[BLOCK_H, BLOCK_W],
        sparsity=SPARSITY,
        trans_A=TRANSPOSE_A,
        trans_B=TRANSPOSE_B,
    )
    os.makedirs(TARGET_PATH, exist_ok=True)
    save_data(A, B, C, idx, TARGET_PATH)
    cmd = os.path.join('.', EXAMPLE_PATH, '99_pit_gemm')
    args = {
        'mode': 'ABC'.index(SPARSE_PORT),
        'M': SHAPE_M,
        'N': SHAPE_N,
        'K': SHAPE_K,
        'bx': BLOCK_W,
        'by': BLOCK_H,
        'sparsity': (idx == 0).sum() / idx.shape[0] / idx.shape[1],
    }
    cmd += ''.join([f' --{k}={v}' for k, v in args.items()])
    with open(os.path.join(TARGET_PATH, 'cmd.txt'), 'w') as f:
        f.write(cmd)
