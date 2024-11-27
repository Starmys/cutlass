/***************************************************************************************************
 * Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <fstream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>
#include "cute/tensor_predicate.hpp"

#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"

__device__ uint32_t float2validx(float val, uint32_t idx) {
  __half half_val = __float2half(val);
  uint32_t bits_val = reinterpret_cast<uint16_t&>(half_val);
  return (idx << 16) | bits_val;
}

__device__ bool compare(uint32_t a, uint32_t b) {
  return *reinterpret_cast<__half*>(&a) < *reinterpret_cast<__half*>(&b);
}

template <class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class GmemTiledCopyA, class SmemTiledCopyA,
          class TB, class BStride, class BSmemLayout, class GmemTiledCopyB, class SmemTiledCopyB,
          class TC, class CStride, class CSmemLayout, class TiledMma>
__global__ static
__launch_bounds__(decltype(size(TiledMma{}))::value)
void gemm_device(
  ProblemShape shape_MNKL, CtaTiler cta_tiler,
  TA const* A, AStride dA, ASmemLayout sA_layout, GmemTiledCopyA copy_gA, SmemTiledCopyA copy_sA,
  TB const* B, BStride dB, BSmemLayout sB_layout, GmemTiledCopyB copy_gB, SmemTiledCopyB copy_sB,
  TC      * C, CStride dC, CSmemLayout          , TiledMma mma
) {
  using namespace cute;

  // Preconditions
  CUTE_STATIC_ASSERT_V(rank(shape_MNKL) == Int<4>{});                  // (M, N, K, TopK)
  CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});                   // (BLK_M, BLK_N, BLK_K)

  CUTE_STATIC_ASSERT_V(size(copy_gA) == size(mma));                     // NumThreads
  CUTE_STATIC_ASSERT_V(size(copy_gB) == size(mma));                     // NumThreads

  static_assert(is_static<ASmemLayout>::value);
  static_assert(is_static<BSmemLayout>::value);
  static_assert(is_static<CSmemLayout>::value);

  CUTE_STATIC_ASSERT_V(size<0>(ASmemLayout{}) == size<0>(cta_tiler));  // BLK_M
  CUTE_STATIC_ASSERT_V(size<0>(CSmemLayout{}) == size<0>(cta_tiler));  // BLK_M
  CUTE_STATIC_ASSERT_V(size<0>(BSmemLayout{}) == size<1>(cta_tiler));  // BLK_N
  CUTE_STATIC_ASSERT_V(size<1>(CSmemLayout{}) == size<1>(cta_tiler));  // BLK_N
  CUTE_STATIC_ASSERT_V(size<1>(ASmemLayout{}) == size<2>(cta_tiler));  // BLK_K
  CUTE_STATIC_ASSERT_V(size<1>(BSmemLayout{}) == size<2>(cta_tiler));  // BLK_K

  CUTE_STATIC_ASSERT_V(congruent(select<0,2>(shape_MNKL), dA));         // dA strides for shape MK
  CUTE_STATIC_ASSERT_V(congruent(select<1,2>(shape_MNKL), dB));         // dB strides for shape NK

  // Represent the full tensors
  Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNKL), dA); // (M,K)
  Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_MNKL), dB); // (N,K)

  // Get the appropriate blocks for this thread block
  // TODO: check gB and gC
  auto cta_coord = make_coord(blockIdx.x, _, 0);                       // (m,n,k)
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});  // (BLK_M,BLK_K)
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});  // (BLK_N,BLK_K,n)
  Tensor gC = make_tensor<float>(select<0,1>(cta_tiler));              // (BLK_M,BLK_N)

  // Compute tile residues for predication
  auto m_max_coord = size<0>(shape_MNKL) - size<0>(gA) * blockIdx.x;   // M - BLK_M * m_coord
  auto n_max_coord = size<1>(shape_MNKL) - size<0>(gB) * blockIdx.y;   // N - BLK_N * n_coord
  // auto residue_mnk = make_tuple(m_max_coord, n_max_coord, 0);

  // Construct shared memory tiles
  extern __shared__ char smem_buf[];
  typedef struct {
    cute::array_aligned<TA, cute::cosize_v<ASmemLayout>> smem_a;
    cute::array_aligned<TB, cute::cosize_v<BSmemLayout>> smem_b;
  } SharedStorage;
  SharedStorage& storage = *((SharedStorage*)smem_buf);
  Tensor sA = make_tensor(make_smem_ptr(storage.smem_a.data()), sA_layout);  // (BLK_M,BLK_K)
  Tensor sB = make_tensor(make_smem_ptr(storage.smem_b.data()), sB_layout);  // (BLK_N,BLK_K,PIPE)

  CUTE_STATIC_ASSERT_V(size<0>(gA) == size<0>(sA));                          // BLK_M
  CUTE_STATIC_ASSERT_V(size<1>(gA) == size<1>(sA));                          // BLK_K
  CUTE_STATIC_ASSERT_V(size<0>(gB) == size<0>(sB));                          // BLK_N
  CUTE_STATIC_ASSERT_V(size<1>(gB) == size<1>(sB));                          // BLK_K
  CUTE_STATIC_ASSERT_V(size<1>(sA) == size<1>(sB));                          // BLK_K

  auto PIPE_MAX = size<2>(sB);

  // Partition the copying of A and B tiles across the threads
  auto gmem_thr_copy_A = copy_gA.get_slice(threadIdx.x);
  auto gmem_thr_copy_B = copy_gB.get_slice(threadIdx.x);

  Tensor tAgA = gmem_thr_copy_A.partition_S(gA);                             // (BCPY,BCPY_M,BCPY_K)
  Tensor tAsA = gmem_thr_copy_A.partition_D(sA);                             // (BCPY,BCPY_M,BCPY_K)
  Tensor tBgB = gmem_thr_copy_B.partition_S(gB);                             // (BCPY,BCPY_N,BCPY_K,n)
  Tensor tBsB = gmem_thr_copy_B.partition_D(sB);                             // (BCPY,BCPY_N,BCPY_K,PIPE)

  //
  // PREFETCH
  //

  // Total count of tiles
  int n_tile_count = size<3>(tBgB);

  // Load A
  copy(copy_gA, tAgA(_,_,_), tAsA(_,_,_));

  // Current tile index in gmem to read from
  auto n_tile_iter = make_coord_iterator(shape<2>(gB));

  // Start async loads for all pipes but the last
  CUTLASS_PRAGMA_UNROLL
  for (int k_pipe = 0; k_pipe < PIPE_MAX-1; ++k_pipe) {
    copy(copy_gB, tBgB(_,_,_,*n_tile_iter), tBsB(_,_,_,k_pipe));
    cp_async_fence();
    --n_tile_count;
    if (n_tile_count > 0) { ++n_tile_iter; }
  }

  //
  // MMA Atom partitioning
  //

  // Tile MMA compute thread partitions and allocate accumulators
  auto thr_mma = mma.get_thread_slice(threadIdx.x);
  Tensor tCrA = thr_mma.partition_fragment_A(sA(_,_));                       // (MMA,MMA_M,MMA_K)
  Tensor tCrB = thr_mma.partition_fragment_B(sB(_,_,0));                     // (MMA,MMA_N,MMA_K)
  Tensor tCgC = thr_mma.partition_C(gC);                                     // (MMA,MMA_M,MMA_N)
  Tensor accum = thr_mma.make_fragment_C(tCgC);                              // (MMA,MMA_M,MMA_N)

  CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(accum));                     // MMA_M
  CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(accum));                     // MMA_N
  CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                      // MMA_K
  CUTE_STATIC_ASSERT_V(size(copy_gA) == size(mma));
  CUTE_STATIC_ASSERT_V(size(copy_gB) == size(mma));

  // CUTE_STATIC_ASSERT_V(size<0>(accum) == _4{});
  // CUTE_STATIC_ASSERT_V(size<1>(accum) == _4{});
  // CUTE_STATIC_ASSERT_V(size<2>(accum) == _8{});
  CUTE_STATIC_ASSERT(size(accum) == 64);

  Tensor new_arr = make_tensor<uint32_t>(make_shape(_16{}, _4{}));
  Tensor max_arr = make_tensor<uint32_t>(make_shape(_16{}, _4{}));

  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < size(max_arr); i++) {
    max_arr(i) = float2validx(-MAXFLOAT, 0);
  }

  //
  // Copy Atom retiling
  //

  auto smem_thr_copy_A   = copy_sA.get_thread_slice(threadIdx.x);
  Tensor tCsA            = smem_thr_copy_A.partition_S(sA);                  // (CPY,CPY_M,CPY_K)
  Tensor tCrA_copy_view  = smem_thr_copy_A.retile_D(tCrA);                   // (CPY,CPY_M,CPY_K)
  CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));            // CPY_M
  CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCrA_copy_view));            // CPY_K

  auto smem_thr_copy_B   = copy_sB.get_thread_slice(threadIdx.x);
  Tensor tCsB            = smem_thr_copy_B.partition_S(sB);                  // (CPY,CPY_N,CPY_K,PIPE)
  Tensor tCrB_copy_view  = smem_thr_copy_B.retile_D(tCrB);                   // (CPY,CPY_N,CPY_K)
  CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // CPY_N
  CUTE_STATIC_ASSERT_V(size<2>(tCsB) == size<2>(tCrB_copy_view));            // CPY_K

  //
  // PIPELINED MAIN LOOP
  //

  // Current pipe index in smem to read from
  int smem_pipe_read  = 0;
  // Current pipe index in smem to write to
  int smem_pipe_write = PIPE_MAX-1;

  Tensor tCsA_p = tCsA(_,_,_);
  Tensor tCsB_p = tCsB(_,_,_,smem_pipe_read);

  // Size of the register pipeline
  auto K_BLOCK_MAX = size<2>(tCrA);

  // PREFETCH register pipeline
  if (K_BLOCK_MAX > 1) {
    // Wait until our first prefetched tile is loaded in
    cp_async_wait<PIPE_MAX-2>();
    __syncthreads();

    // Prefetch the first rmem from the first k-tile
    copy(copy_sA, tCsA_p(_,_,Int<0>{}), tCrA_copy_view(_,_,Int<0>{}));
    copy(copy_sB, tCsB_p(_,_,Int<0>{}), tCrB_copy_view(_,_,Int<0>{}));
  }

  // float max_val = -MAXFLOAT;
  // float max_pos = 0.0f;
  const uint32_t warp_id = threadIdx.x / 32;
  const uint32_t lane_id = threadIdx.x % 32;

  CUTLASS_PRAGMA_NO_UNROLL
  for (int n_tile_idx = 0; n_tile_count > -(PIPE_MAX-1); n_tile_idx++)
  {
    clear(accum);

    // Pipeline the outer products with a static for loop.
    //
    // Note, the for_each() function is required here to ensure `k_block` is of type Int<x>.
    for_each(make_int_sequence<K_BLOCK_MAX>{}, [&] (auto k_block)
    {
      if (k_block == K_BLOCK_MAX - 1)
      {
        // Slice the smem_pipe_read smem
        tCsA_p = tCsA(_,_,_);
        tCsB_p = tCsB(_,_,_,smem_pipe_read);

        // Commit the smem for smem_pipe_read
        cp_async_wait<PIPE_MAX-2>();
        __syncthreads();
      }

      // Load A, B shmem->regs for k_block+1
      auto k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX;  // static
      copy(copy_sA, tCsA_p(_,_,k_block_next), tCrA_copy_view(_,_,k_block_next));
      copy(copy_sB, tCsB_p(_,_,k_block_next), tCrB_copy_view(_,_,k_block_next));
      // Copy gmem to smem before computing gemm on each k-pipe
      if (k_block == 0)
      {
        copy(copy_gB, tBgB(_,_,_,*n_tile_iter), tBsB(_,_,_,smem_pipe_write));
        cp_async_fence();
        if (--n_tile_count > 0) { ++n_tile_iter; }

        // Advance the pipe -- Doing it here accounts for K_BLOCK_MAX = 1 (no rmem pipe)
        smem_pipe_write = smem_pipe_read;
        ++smem_pipe_read;
        smem_pipe_read = (smem_pipe_read == PIPE_MAX) ? 0 : smem_pipe_read;
      }

      // Thread-level register gemm for k_block
      gemm(mma, accum, tCrA(_,_,k_block), tCrB(_,_,k_block), accum);
    });

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(accum); i++) {
      new_arr(i) = float2validx(accum(i), n_tile_idx * size<1>(cta_tiler) + i);  // TODO
      // printf("[Thread #%d] new_arr(%d)=(%f) => (%d)\n", threadIdx.x, i, __half2float(accum(i)), (unsigned int)(new_arr(i)));
    }

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 16; i++) {  // Loop for rows
      // Warp Sort (NewArr)
      if (compare(new_arr(i, 2), new_arr(i, 0))) {
        auto tmp = new_arr(i, 0);
        new_arr(i, 0) = new_arr(i, 2);
        new_arr(i, 2) = tmp;
      }
      if (compare(new_arr(i, 1), new_arr(i, 0))) {
        auto tmp = new_arr(i, 0);
        new_arr(i, 0) = new_arr(i, 1);
        new_arr(i, 1) = tmp;
      }
      if (compare(new_arr(i, 3), new_arr(i, 2))) {
        auto tmp = new_arr(i, 2);
        new_arr(i, 2) = new_arr(i, 3);
        new_arr(i, 3) = tmp;
      }
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < 4; j++) {
        CUTLASS_PRAGMA_UNROLL
        for (int lane_mask = 16; lane_mask > 0; lane_mask >>= 1) {
          uint32_t x = __shfl_xor_sync(0xffffffff, new_arr(i, j), lane_mask);
          if (compare(x, new_arr(i, j)) ^ ((lane_id ^ lane_mask) < lane_id)) {
            new_arr(i, j) = x;
          }
        }
      }
      // Merge (Overwrite MaxArr)
      if (compare(max_arr(i, 0), new_arr(i, 0))) {
        max_arr(i, 0) = new_arr(i, 0);
      }
      if (compare(max_arr(i, 1), new_arr(i, 1))) {
        max_arr(i, 1) = new_arr(i, 1);
      }
      if (compare(max_arr(i, 2), new_arr(i, 2))) {
        max_arr(i, 2) = new_arr(i, 2);
      }
      if (compare(max_arr(i, 3), new_arr(i, 3))) {
        max_arr(i, 3) = new_arr(i, 3);
      }
      // Warp Sort (MaxArr)
      if (compare(max_arr(i, 0), max_arr(i, 2))) {
        auto tmp = max_arr(i, 0);
        max_arr(i, 0) = max_arr(i, 2);
        max_arr(i, 2) = tmp;
      }
      if (compare(max_arr(i, 0), max_arr(i, 1))) {
        auto tmp = max_arr(i, 0);
        max_arr(i, 0) = max_arr(i, 1);
        max_arr(i, 1) = tmp;
      }
      if (compare(max_arr(i, 2), max_arr(i, 3))) {
        auto tmp = max_arr(i, 2);
        max_arr(i, 2) = max_arr(i, 3);
        max_arr(i, 3) = tmp;
      }
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < 4; j++) {
        CUTLASS_PRAGMA_UNROLL
        for (int lane_mask = 16; lane_mask > 0; lane_mask >>= 1) {
          uint32_t x = __shfl_xor_sync(0xffffffff, max_arr(i, j), lane_mask);
          if (compare(x, max_arr(i, j)) ^ (lane_id < (lane_id ^ lane_mask))) {
            max_arr(i, j) = x;
          }
        }
      }
    }

    // CUTLASS_PRAGMA_UNROLL
    // for (int i = 0; i < size(accum); ++i) {
    //   // Max
    //   if (accum(i) > max_val) {
    //     max_val = accum(i);
    //     max_pos = n_tile_idx * size<1>(cta_tiler) + i;
    //   }

    //   // Row => Thread
    //   // auto cnt = atomicAdd(
    //   //   reinterpret_cast<half*>(&C[int(accum(i)) * 320 + 319]),
    //   //   __float2half(1.0f)
    //   // );
    //   // C[int(accum(i)) * 320 + int(cnt)] = threadIdx.x;

    //   // Row, Col => Thread
    //   // C[int(accum(i)) * size<0>(dC) + int(cnt)] = threadIdx.x;
    // }
  }

  cp_async_wait<0>();
  __syncthreads();

  //
  // Epilogue
  //

  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < 16; i++) {
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < 4; j++) {
      C[  // TODO
        (blockIdx.x * size<0>(cta_tiler) + warp_id * 16 + i) * size<0>(dC)
        +
        j * 32 + lane_id
      ] = max_arr(i, j);
    }
  }

  // Thread => Max
  // C[(blockIdx.x * size<0>(cta_tiler) + threadIdx.x % size<0>(cta_tiler)) * size<0>(dC) + 0] = max_val;
  // C[(blockIdx.x * size<0>(cta_tiler) + threadIdx.x % size<0>(cta_tiler)) * size<0>(dC) + 1] = max_pos;
}

// Setup params for a NT GEMM
template <class TA, class TB, class TC>
void gemm_nt(
  int m, int n, int k, int l,
  TA const* A, int ldA,
  TB const* B, int ldB,
  TC      * C, int ldC,
  cudaStream_t stream = 0
) {
  using namespace cute;

  // Define shapes (dynamic)
  auto M = int(m);
  auto N = int(n);
  auto K = int(k);
  auto L = int(l);
  auto prob_shape = make_shape(M, N, K, L);                  // (M, N, K, L)

  // Define NT strides (mixed)
  auto dA = make_stride(ldA, Int<1>{});                      // (dM, dK)
  auto dB = make_stride(ldB, Int<1>{});                      // (dN, dK)
  auto dC = make_stride(ldC, Int<1>{});                      // (dM, dL)

  // Define CTA tile sizes (static)
  auto bM = Int<64>{};
  auto bN = Int<128>{};
  auto bK = Int<128>{};
  auto cta_tiler = make_shape(bM, bN, bK);                   // (BLK_M, BLK_N, BLK_K)
  auto bP = Int<2>{};  // Pipeline

  // Define the smem layouts (static)
  // auto sA = make_layout(make_shape(bM, bK)); // TODO
  auto sA = tile_to_shape(
    composition(Swizzle<3, 3, 3>{}, Layout<Shape<_8, _64>, Stride<_64, _1>>{}),
    make_shape(bM, bK)
  );
  auto sB = tile_to_shape(
    composition(Swizzle<3, 3, 3>{}, Layout<Shape<_8, _64>, Stride<_64, _1>>{}),
    make_shape(bN, bK, bP)
  );
  auto sC = make_layout(make_shape(bM, bN));
  static_assert(rank(sB) == 3, "B Smem layout must be rank 3.");
  int const smem_bytes = cosize_v<decltype(sA)> * sizeof(TA) + cosize_v<decltype(sB)> * sizeof(TB);

  // Define the thread layouts (static)

  TiledCopy copyGA = make_tiled_copy(Copy_Atom<UniversalCopy<TA>, TA>{},
                                    Layout<Shape<_16,_8>, Stride<_8,_1>>{},
                                    Layout<Shape< _1,_8>>{});
  TiledCopy copyGB = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, TB>{},
                                    Layout<Shape<_16,_8>, Stride<_8,_1>>{},
                                    Layout<Shape< _1,_8>>{});

  TiledMMA mmaC = make_tiled_mma(SM80_16x8x16_F32F16F16F32_TN{},
                                 Layout<Shape<_4,_1,_1>>{},
                                 Tile<_64,_64,_16>{});

  auto copySA = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, TA>{}, mmaC);
  auto copySB = make_tiled_copy_B(Copy_Atom<SM75_U32x4_LDSM_N, TB>{}, mmaC);

#if 0
  print(copyGA);
  print(copyGB);
  print(copySA);
  print(copySB);
  print(mmaC);
  print(sA);
  print(sB);
#endif

  dim3 dimBlock(size(mmaC));
  dim3 dimGrid(size(ceil_div(M, bM)), 1, 1);

  if (smem_bytes >= (48 << 10)) {
    cudaFuncSetAttribute(
      gemm_device<
        decltype(prob_shape), decltype(cta_tiler),
        TA, decltype(dA), decltype(sA), decltype(copyGA), decltype(copySA),
        TB, decltype(dB), decltype(sB), decltype(copyGB), decltype(copySB),
        TC, decltype(dC), decltype(sC), decltype(mmaC)
      >,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      smem_bytes
    );
  }

  gemm_device<<<dimGrid, dimBlock, smem_bytes, stream>>>(
    prob_shape, cta_tiler,
    A, dA, sA, copyGA, copySA,
    B, dB, sB, copyGB, copySB,
    C, dC, sC, mmaC
  );
}

template<class ElementIn, class ElementOut>
int load_array_from_file(ElementOut* arr, std::string filepath) {
  std::cout << "Loading: " << filepath << std::endl;
  std::ifstream infile(filepath);
  int length = 0;
  ElementIn current_number;
  while (infile >> current_number) {
    arr[length++] = current_number;
  }
  return length;
}

template<class ElementIn, class ElementOut>
void print_tensor(ElementIn* arr, int rows, int cols) {
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      std::cout << ElementOut(arr[i * cols + j]) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

int main(int argc, char** argv)
{
  cudaDeviceProp props;
  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (error != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
    return -1;
  }

  if (props.major < 8) {
    std::cout << "This example requires an Ampere GPU or newer (CC >= 80)" << std::endl;
    // Return 0 so tests pass if run on unsupported architectures or CUDA Toolkits.
    return 0;
  }

  int m = 5120;
  if (argc >= 2)
    sscanf(argv[1], "%d", &m);

  int n = 5120;
  if (argc >= 3)
    sscanf(argv[2], "%d", &n);

  int k = 4096;
  if (argc >= 4)
    sscanf(argv[3], "%d", &k);

  int l = 8192;
  if (argc >= 5)
    sscanf(argv[4], "%d", &l);

  char tmp_path[1024];
  if (argc >= 6)
    sscanf(argv[5], "%1024s", &tmp_path);
  std::string data_folder = tmp_path;

  using TA = cute::half_t;
  using TB = cute::half_t;
  using TC = uint32_t;

  std::cout << "M = " << m << std::endl;
  std::cout << "N = " << n << std::endl;
  std::cout << "K = " << k << std::endl;
  std::cout << "L = " << l << std::endl;
  std::cout << "C = A B^T" << std::endl;

  thrust::host_vector<TA> h_A(m * k);
  thrust::host_vector<TB> h_B(n * k);
  thrust::host_vector<TC> h_C(m * l);
  thrust::host_vector<TC> h_C_ref(m * l);

  if (data_folder.length() > 0) {
    load_array_from_file<float, TA>(h_A.data(), data_folder + "/A.txt");
    load_array_from_file<float, TB>(h_B.data(), data_folder + "/B.txt");
    // load_array_from_file<int, TC>(h_C_ref.data(), data_folder + "/C.txt");
  } else {
    for (int j = 0; j < m * k; ++j) h_A[j] = static_cast<TA>( 2 * (rand() / double(RAND_MAX)) - 1 );
    for (int j = 0; j < n * k; ++j) h_B[j] = static_cast<TB>( 2 * (rand() / double(RAND_MAX)) - 1 );
  }
  for (int j = 0; j < m * l; ++j) h_C[j] = static_cast<TC>( 0 );

  thrust::device_vector<TA> d_A = h_A;
  thrust::device_vector<TB> d_B = h_B;
  thrust::device_vector<TC> d_C = h_C;

  // Run once
  d_C = h_C;
  gemm_nt(
    m, n, k, l,
    d_A.data().get(), k,
    d_B.data().get(), k,
    d_C.data().get(), l
  );
  CUTE_CHECK_LAST();
  thrust::host_vector<TC> cute_result = d_C;

  // Check correctness
  // if (data_folder.length() > 0) {
  //   double diff = 0.0;
  //   for (int j = 0; j < l; ++j) {
  //     double tmp_diff = 0.0;
  //     for (int i = 0; i < n; ++i) {
  //       tmp_diff += abs(cute_result[j * n + i] - h_C_ref[j * n + i]);
  //     }
  //     diff += tmp_diff / n;
  //   }
  //   diff /= m;
  //   // passed = diff / k < 1e-6;
  //   printf("Diff: %f\n", diff);
  // }

  // printf("In:\n");
  // print_tensor<TA, float>(h_A.data(), m, k);
  // printf("Weight:\n");
  // print_tensor<TB, float>(h_B.data(), n, k);
  // printf("Out:\n");
  // print_tensor<TC, unsigned int>(cute_result.data(), m, l);
  // printf("Ref:\n");
  // print_tensor<TC, float>(h_C_ref.data(), m, l);

  double gflops = (2.0*m*n*k) * 1e-9;

  const int timing_iterations = 100;
  GPU_Clock timer;

  // Timing iterations
  for (int i = 0; i < timing_iterations; ++i) {
    gemm_nt(
      m, n, k, l,
      d_A.data().get(), k,
      d_B.data().get(), k,
      d_C.data().get(), l
    );
  }
  timer.start();
  for (int i = 0; i < timing_iterations; ++i) {
    gemm_nt(
      m, n, k, l,
      d_A.data().get(), k,
      d_B.data().get(), k,
      d_C.data().get(), l
    );
  }
  double cute_time = timer.seconds() / timing_iterations;
  CUTE_CHECK_LAST();
  printf("CUTE_GEMM:     [%6.1f]GFlop/s  (%6.4f)ms\n", gflops / cute_time, cute_time*1000);

  return 0;
}
