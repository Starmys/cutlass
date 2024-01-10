import os

import torch
from scipy.sparse import csc_array, csr_array, coo_array
from sparta.testing import profile

import quant_cuda


EXAMPLE_PATH = os.path.join('examples', '98_squeeze_llm')
TARGET_PATH = os.path.join(
    os.path.dirname(__file__).replace(EXAMPLE_PATH, os.path.join('build', EXAMPLE_PATH)),
    'data'
)

BATCH_SIZE = 1
IN_FEATURES = 8192  # 4096
OUT_FEATURES = 8192  # 4096
W_BITS = 4
SPARSITY = 0.0045
TOP_X = 10

BLOCK_SIZE_OUT = 64
BLOCK_SIZE_IN = 256
NUM_BLOCKS_OUT = OUT_FEATURES // BLOCK_SIZE_OUT
NUM_BLOCKS_IN = OUT_FEATURES // BLOCK_SIZE_IN
ROWS = OUT_FEATURES * (IN_FEATURES // BLOCK_SIZE_IN)

WARP_SIZE = 1 << W_BITS
NUM_WARPS = BLOCK_SIZE_IN // WARP_SIZE
BLOCK_SIZE_Q = BLOCK_SIZE_IN * W_BITS // 32
ACCESS_SIZE = 4

sqllm_kernel_name = 'vecquant4matmul_spmv_hybrid_nuq_perchannel'
if BATCH_SIZE > 1:
    sqllm_kernel_name += '_batched'
sqllm_hybrid_kernel = getattr(quant_cuda, sqllm_kernel_name)
sqllm_quant_kernel = getattr(quant_cuda, sqllm_kernel_name.replace('_spmv_hybrid', ''))


def generate_data(dtype: torch.dtype = torch.float32):
    activations = torch.randn((BATCH_SIZE, IN_FEATURES), dtype=dtype, device='cpu')
    quant_weight = torch.randint(
        -0x80000000, 0x80000000,
        (IN_FEATURES // 32 * W_BITS, OUT_FEATURES),
        dtype=torch.int32, device='cpu',
    )
    # quant_weight = torch.zeros((IN_FEATURES // 32 * W_BITS, OUT_FEATURES), dtype=torch.int32, device='cpu')
    quant_lut = torch.randn((OUT_FEATURES, 2 ** W_BITS), dtype=dtype, device='cpu')

    sparse_nnz = round(IN_FEATURES * OUT_FEATURES * SPARSITY)
    sparse_weight = torch.randn((sparse_nnz, ), dtype=torch.float32, device='cpu')
    # sparse_weight = torch.zeros((sparse_nnz, ), dtype=torch.float32, device='cpu')
    sparse_rows = torch.randint(0, IN_FEATURES, (sparse_nnz, ), dtype=torch.int32, device='cpu')
    sparse_cols = torch.randint(0, OUT_FEATURES, (sparse_nnz, ), dtype=torch.int32, device='cpu')

    # full_weight = torch.randn((IN_FEATURES, TOP_X), dtype=dtype, device='cpu')
    full_weight = torch.zeros((IN_FEATURES, TOP_X), dtype=dtype, device='cpu')
    full_cols = torch.randint(0, OUT_FEATURES, (TOP_X, ), dtype=torch.int32, device='cpu')

    return {
        'activations': activations,
        'quant_weight': quant_weight,
        'quant_lut': quant_lut,
        'sparse_weight': sparse_weight,
        'sparse_rows': sparse_rows,
        'sparse_cols': sparse_cols,
        'full_weight': full_weight,
        'full_cols': full_cols,
    }


def to_sqllm_inputs(data: dict[str, torch.Tensor]):
    dtype = data['activations'].dtype
    sparse_weight = data['sparse_weight'].cpu().numpy()
    sparse_rows = data['sparse_rows'].cpu().numpy()
    sparse_cols = data['sparse_cols'].cpu().numpy()
    sparse_arr = csc_array((sparse_weight, (sparse_rows, sparse_cols)), shape=(IN_FEATURES, OUT_FEATURES))
    return [
        torch.tensor(sparse_arr.indptr[:IN_FEATURES + 1], dtype=torch.int32, device='cuda'),
        torch.tensor(sparse_arr.indices, dtype=torch.int32, device='cuda'),
        torch.tensor(sparse_arr.data, dtype=dtype, device='cuda'),
        data['activations'].cuda(),
        data['full_weight'].cuda(),
        data['full_cols'].cuda(),
        data['quant_weight'].cuda(),
        data['quant_lut'].cuda(),
    ]


def sqllm_quant(
    sparse_rows: torch.Tensor,
    sparse_cols: torch.Tensor,
    sparse_weight: torch.Tensor,
    activations: torch.Tensor,
    full_weight: torch.Tensor,
    full_cols: torch.Tensor,
    quant_weight: torch.Tensor,
    quant_lut: torch.Tensor,
):
    outputs = torch.zeros((BATCH_SIZE, OUT_FEATURES), dtype=torch.float32, device='cuda')
    sqllm_quant_kernel(
        activations,
        quant_weight,
        outputs,
        quant_lut,
    )
    return outputs


def sqllm_hybrid(
    sparse_rows: torch.Tensor,
    sparse_cols: torch.Tensor,
    sparse_weight: torch.Tensor,
    activations: torch.Tensor,
    full_weight: torch.Tensor,
    full_cols: torch.Tensor,
    quant_weight: torch.Tensor,
    quant_lut: torch.Tensor,
):
    outputs = torch.zeros((BATCH_SIZE, OUT_FEATURES), dtype=torch.float32, device='cuda')
    sqllm_hybrid_kernel(
        sparse_rows,
        sparse_cols,
        sparse_weight,
        activations,
        full_weight,
        full_cols,
        outputs,
        OUT_FEATURES,
        quant_weight,
        quant_lut,
    )
    return outputs


def to_torch_inputs(data: dict[str, torch.Tensor]):
    activations = data['activations'].cuda()
    quant_weight = torch.stack([
        data['quant_weight'].bitwise_right_shift(offset).bitwise_and(0xf)
        for offset in range(0, 32, 4)
    ]).swapaxes(0, 1).reshape((IN_FEATURES, OUT_FEATURES))
    weight = torch.stack([l[q] for l, q in zip(data['quant_lut'], quant_weight.T)]).T
    sparse_weight = data['sparse_weight'].cpu().numpy()
    sparse_rows = data['sparse_rows'].cpu().numpy()
    sparse_cols = data['sparse_cols'].cpu().numpy()
    sparse_arr = coo_array((sparse_weight, (sparse_rows, sparse_cols)), shape=(IN_FEATURES, OUT_FEATURES))
    weight += torch.tensor(sparse_arr.toarray(), dtype=weight.dtype)
    for col_idx, col_val in zip(data['full_cols'], data['full_weight'].T):
        weight[:, col_idx] += col_val
    return [activations.cuda(), weight.cuda()]


def torch_ref(activations: torch.Tensor, weight: torch.Tensor):
    return torch.mm(activations, weight)


def to_cutlass_inputs(data: dict[str, torch.Tensor], ref_output: torch.Tensor):
    quant_weight = torch.stack([
        data['quant_weight'].bitwise_right_shift(offset).bitwise_and((1 << W_BITS) - 1)
        for offset in range(0, 32, W_BITS)
    ]).swapaxes(0, 1).reshape((IN_FEATURES, OUT_FEATURES)).T  # [OUT_FEATURES, IN_FEATURES]
    quant_weight = quant_weight.reshape((
        NUM_BLOCKS_OUT, BLOCK_SIZE_OUT, NUM_BLOCKS_IN, BLOCK_SIZE_IN
    )).swapaxes(1, 2).reshape((-1, WARP_SIZE, NUM_WARPS, WARP_SIZE))
    quant_weight = quant_weight.swapaxes(1, 3).reshape((-1, WARP_SIZE, BLOCK_SIZE_IN))
    quant_weight = torch.stack([
        quant_weight[:, :, i::(32 // W_BITS)].bitwise_left_shift(offset)
        for i, offset in enumerate(range(0, 32, W_BITS))
    ]).sum(dim=0)
    quant_weight = quant_weight.reshape((
        -1, WARP_SIZE, BLOCK_SIZE_Q // ACCESS_SIZE, ACCESS_SIZE
    )).swapaxes(1, 2)
    # quant_weight = data['quant_weight'].T
    sparse_weight = data['sparse_weight'].cpu().numpy()
    sparse_rows = data['sparse_rows'].cpu().numpy()
    sparse_cols = data['sparse_cols'].cpu().numpy()
    sparse_arr = coo_array((sparse_weight, (sparse_rows, sparse_cols)), shape=(IN_FEATURES, OUT_FEATURES))
    sparse_arr = torch.tensor(sparse_arr.toarray())
    for col_idx, col_val in zip(data['full_cols'], data['full_weight'].T):
        sparse_arr[:, col_idx] += col_val
    sparse_arr = sparse_arr.T.reshape((
        NUM_BLOCKS_OUT, BLOCK_SIZE_OUT, NUM_BLOCKS_IN, BLOCK_SIZE_IN
    )).swapaxes(1, 2).reshape((ROWS, BLOCK_SIZE_IN))
    sparse_arr = csr_array(sparse_arr, shape=(ROWS, BLOCK_SIZE_IN))
    return {
        'activations': data['activations'],
        # 'weight': torch_inputs[1].T,
        'quant_weight': quant_weight,
        'quant_lut': data['quant_lut'],
        'sparse_weight': torch.tensor(sparse_arr.data, dtype=data['sparse_weight'].dtype, device='cuda'),
        'sparse_row': torch.tensor(sparse_arr.indptr[:ROWS + 1], dtype=torch.int32, device='cuda'),
        'sparse_col': torch.tensor(sparse_arr.indices, dtype=torch.int32, device='cuda'),
        'output': ref_output,
    }


def save_tensor(tensor: torch.Tensor, path: str):
    if 'float' in str(tensor.dtype):
        tensor = tensor.to(torch.float32)
        format_str = '%.6f'
    else:
        tensor = tensor.to(torch.int32)
        format_str = '%d'
    array = tensor.flatten().cpu().numpy()
    with open(path, 'w') as f:
        f.write(' '.join(map(lambda x: format_str % x, array)))


def save_data(
    data: dict[str, torch.Tensor],
    path: str,
):
    for name, val in data.items():
        save_tensor(val, os.path.join(path, f'{name}.txt'))


if __name__ == '__main__':
    torch.manual_seed(2023)

    data = generate_data(torch.float32)

    torch_inputs = to_torch_inputs(data)
    torch_output = torch_ref(*torch_inputs)

    sqllm_inputs = to_sqllm_inputs(data)
    sqllm_output = sqllm_hybrid(*sqllm_inputs)

    torch.testing.assert_close(torch_output, sqllm_output, rtol=1e-4, atol=1e-4)

    torch_inputs_fp16 = [x.to(torch.float16) for x in torch_inputs]
    print(f'PyTorch Dense (FP16): {profile(torch_ref, torch_inputs_fp16):.6f} ms')
    print(f'PyTorch Dense (FP32): {profile(torch_ref, torch_inputs):.6f} ms')
    print(f'  SQLLM Quant       : {profile(sqllm_quant, sqllm_inputs):.6f} ms')
    print(f'  SQLLM Hybrid      : {profile(sqllm_hybrid, sqllm_inputs):.6f} ms')

    os.makedirs(TARGET_PATH, exist_ok=True)

    cutlass_inputs = to_cutlass_inputs(data, torch_output)
    save_data(cutlass_inputs, TARGET_PATH)

    cmd = os.path.join('.', EXAMPLE_PATH, '98_squeeze_llm')
    args = {
        'batch': BATCH_SIZE,
        'N': OUT_FEATURES,
        'K': IN_FEATURES,
        'nnz': cutlass_inputs['sparse_row'][-1].item(),
        'rows': ROWS,
    }
    cmd += ''.join([f' --{k}={v}' for k, v in args.items()])
    with open(os.path.join(TARGET_PATH, 'cmd.txt'), 'w') as f:
        f.write(cmd)
