// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#define MAX_BLOCK_THREAD_COUNT 1024

__device__ __forceinline__ const unsigned char* add_ptr_b(const unsigned char* src, int offset) \
{                                                                                               \
    const unsigned char* dst;                                                                   \
    asm("{                       \n\t"                                                          \
        ".reg .u32  lo, hi, of;  \n\t"                                                          \
        "mul.lo.u32 of, %2, %3;  \n\t"                                                          \
        "mov.b64    {lo,hi},%1;  \n\t"                                                          \
        "add.cc.u32 lo, lo, of;  \n\t"                                                          \
        "addc.u32   hi, hi, 0;   \n\t"                                                          \
        "mov.b64    %0,{lo,hi};  \n\t"                                                          \
        "}" : "=l"(dst) : "l"(src), "r"(offset), "r"((int)sizeof(*src)));                       \
    return dst;                                                                                 \
}

template <
    int blockHeight,
    int blockWidth
>
__global__ void bcs_index_1(
    const unsigned char * __restrict__ mask,
    int * row_ptr,
    int * extra_buffer,
    int H,
    int W
) {
    assert(blockDim.x <= MAX_BLOCK_THREAD_COUNT);
    // Initialize the shared flag
    __shared__ unsigned char reduce[MAX_BLOCK_THREAD_COUNT];

    uint bx = blockIdx.x;
    uint by = blockIdx.y;
    uint tid = threadIdx.x;

    uint global_offset = (by * blockHeight) * W + bx * blockWidth;
    uint threadSize = blockHeight * blockWidth / 16;
    assert(threadSize % blockDim.x == 0);

    uint flag = 0;
    for (uint _pos = tid; _pos < threadSize; _pos += blockDim.x) {
        uint block_offset = _pos / (blockWidth / 16) * W + _pos % (blockWidth / 16) * 16;
        uint4 data = __ldg((const uint4*)(add_ptr_b(mask, global_offset + block_offset)));
        flag = (flag || data.x || data.y || data.z || data.w);
    }
    reduce[tid] = flag;

    // Fast tree reduce accross the block
    __syncthreads();
    for (uint s = blockDim.x >> 1; s > 32; s >>= 1) {
        if (tid < s) reduce[tid] = (reduce[tid] || reduce[tid + s]);
        __syncthreads();
    }
    #pragma unroll
    for (uint s = 32; s > 1; s >>= 1) {
        reduce[tid] += reduce[tid + s];
    }
    __syncthreads();

    if (tid == 0 && reduce[0] > 0) {
        // Record BCSR column index, +1 because 0 means empty
        int col_pos_id = atomicAdd(&extra_buffer[by], 1);
        extra_buffer += gridDim.y;
        extra_buffer[gridDim.x * by + col_pos_id] = bx + 1;
        // Record pointers
        atomicAdd(&row_ptr[by + 1], 1);
    }
}

__global__ void bcs_index_2(
    int * row_ptr,
    int * BCSR_idx,
    int * extra_buffer
) {
    uint bx = blockIdx.x;
    uint by = blockIdx.y;
    extra_buffer += gridDim.y;
    int cidx = extra_buffer[gridDim.x * by + bx];
    if (cidx > 0) {
        BCSR_idx[row_ptr[by] + bx] = (by << 16) + cidx - 1;
    }
}
