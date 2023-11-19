
# %%
# Benchmark
# ---------
#
# Square Matrix Performance
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We can now compare the performance of our kernel against that of cuBLAS. Here we focus on square matrices,
# but feel free to arrange this script as you wish to benchmark any other matrix shape.


import torch
import triton

from generate_test_data import generate_data
from triton_dense_matmul import triton_dense_matmul
from triton_pit_matmul import triton_pit_matmul


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M', 'N', 'K'],  # Argument names to use as an x-axis for the plot
        x_vals=[
            1024, 2048, 4096,
            # 128 * i for i in range(2, 33)
        ],  # Different possible values for `x_name`
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot
        # Possible values for `line_arg`
        line_vals=['cublas', 'triton-dense', 'triton-pit'],
        # Label name for the lines
        line_names=["cuBLAS", "Triton (Dense)", "Triton (PIT)"],
        # Line styles
        styles=[('green', '-'), ('blue', '-'), ('red', '-')],
        ylabel="TFLOPS",  # Label name for the y-axis
        plot_name="matmul-performance",  # Name for the plot, used also as a file name for saving the plot.
        args={},
    )
)
def benchmark(M, N, K, provider):
    torch.manual_seed(2023)
    A, B, C, idx = generate_data(
        sparse_port='B',
        shape=[M, K, N],
        block=[1, 32],
        sparsity=0.95,
        trans_A=False,
        trans_B=False,
    )
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'cublas':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(A, B), quantiles=quantiles)
    elif provider == 'triton-dense':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_dense_matmul(A, B), quantiles=quantiles)
    elif provider == 'triton-pit':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_pit_matmul(A, B, idx), quantiles=quantiles)
    # perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    # return perf(ms), perf(max_ms), perf(min_ms)
    return ms, min_ms, max_ms


benchmark.run(show_plots=True, print_data=True)