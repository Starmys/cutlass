import os
from typing import Optional, Callable, List

import torch


EXAMPLE_PATH = os.path.join('examples', '97_pit_gemv')
TARGET_PATH = os.path.join(
    os.path.dirname(__file__).replace(EXAMPLE_PATH, os.path.join('build', EXAMPLE_PATH)),
    'data'
)

BATCH_SIZE = 1
IN_FEATURES = 13824
OUT_FEATURES = 5120
SELECTED_ROWS = 5000
# SELECTED_ROWS = 13824

# IN_FEATURES = 1024
# OUT_FEATURES = 32768
# SELECTED_ROWS = 1024


def profile(
    func: Callable,
    inputs: List,
    target_outputs: Optional[List] = None,
    num_warmups: int = 200,
    num_iters: int = 1000,
    cuda: bool = False,
) -> float:
    if target_outputs is not None:
        check(func, inputs, target_outputs)
    torch.cuda.synchronize()
    for _ in range(num_warmups):
        func(*inputs)
    torch.cuda.synchronize()
    if cuda:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
            for _ in range(num_iters):
                func(*inputs)
        latency = 0
        for event in p.key_averages():
            if event.key != 'cudaDeviceSynchronize':
                latency += event.cuda_time * event.count
        latency /= num_iters * 1000
    else:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(num_iters):
            func(*inputs)
        end.record()
        torch.cuda.synchronize()
        latency = start.elapsed_time(end) / num_iters
    return latency


def check(func: Callable, inputs: List, target_outputs: List):
    outputs = func(*inputs)
    if len(target_outputs) == 1:
        outputs = [outputs]
    assert len(outputs) == len(target_outputs), f'expected {len(target_outputs)} outputs, got {len(outputs)}'
    for output, target_output in zip(outputs, target_outputs):
        torch.testing.assert_close(output, target_output, atol=1e-4, rtol=1e-4)


def generate_data(dtype: torch.dtype = torch.float32):
    weight = torch.randn((IN_FEATURES, OUT_FEATURES), dtype=dtype, device='cpu')
    activations = torch.randn((BATCH_SIZE, IN_FEATURES), dtype=dtype, device='cpu')
    # weight = torch.ones((IN_FEATURES, OUT_FEATURES), dtype=dtype, device='cpu')
    # activations = torch.ones((BATCH_SIZE, IN_FEATURES), dtype=dtype, device='cpu')

    rows = torch.arange(IN_FEATURES, dtype=torch.int32, device='cpu')[torch.randperm(IN_FEATURES)]
    rows = rows.unsqueeze(0).repeat((BATCH_SIZE, 1))

    return {
        'weight': weight,
        'activations': activations,
        'rows': rows,
    }


def to_torch_inputs(data: dict[str, torch.Tensor]):
    activations = data['activations'].clone().cuda()
    weight = data['weight'].cuda()
    rows = data['rows'].cuda()
    for batch_idx in range(BATCH_SIZE):
        activations[batch_idx, rows[batch_idx, SELECTED_ROWS:]] = 0
    return [activations, weight]


def torch_ref(activations: torch.Tensor, weight: torch.Tensor):
    return torch.mm(activations, weight)


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

    torch_inputs_fp16 = [x.to(torch.float16) for x in torch_inputs]
    print(f'PyTorch Dense (FP16): {profile(torch_ref, torch_inputs_fp16):.6f} ms')
    print(f'PyTorch Dense (FP32): {profile(torch_ref, torch_inputs):.6f} ms')

    os.makedirs(TARGET_PATH, exist_ok=True)

    data['output'] = torch_output
    
    rows = data['rows']
    rows[:, :SELECTED_ROWS], _ = rows[:, :SELECTED_ROWS].sort()
    data['rows'] = rows

    save_data(data, TARGET_PATH)

    cmd = os.path.join('.', EXAMPLE_PATH, '97_pit_gemv')
    args = {
        'batch': BATCH_SIZE,
        'N': OUT_FEATURES,
        'K': IN_FEATURES,
        'rows': SELECTED_ROWS,
    }
    cmd += ''.join([f' --{k}={v}' for k, v in args.items()])
    with open(os.path.join(TARGET_PATH, 'cmd.txt'), 'w') as f:
        f.write(cmd)
