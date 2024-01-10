/***************************************************************************************************
 * Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/*! \file
    \brief 
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/complex.h"
#include "cutlass/tensor_ref.h"

#include "cutlass/arch/memory.h"
#include "cutlass/arch/cache_operation.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/layout/matrix.h"

#include "cutlass/numeric_conversion.h"
/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename ElementA_,
  typename LayoutA_,
  typename ElementB_,
  typename ElementC_,
  typename ElementAccumulator_,
  int kElementsPerAccess_ = 1,            ///< Number of elements involved in a global access.
  int kThreadCount_ = 0,                  ///< Number of threads in the thread block.
                                          ///  It will be calculated automatically if set to 0.
  int kStride_ = 0
>
struct SqueezeLLMGemv;

// GEMV for row-major A matrix
template <
    typename ElementA_,
    typename ElementB_,
    typename ElementC_,
    typename ElementAccumulator_,
    int kElementsPerAccess_,
    int kThreadCount_,
    int kStride_ 
>
struct SqueezeLLMGemv <
    ElementA_,            
    layout::RowMajor,
    ElementB_,            
    ElementC_,
    ElementAccumulator_,
    kElementsPerAccess_,
    kThreadCount_,
    kStride_
>{
public:

  using ElementQ = int;
  using LayoutA = layout::RowMajor;
  using TensorRefA = TensorRef<ElementQ, LayoutA>;

  using ElementA = ElementA_;
  using ElementB = ElementB_;
  using ElementC = ElementC_;

  using ElementAccumulator = ElementAccumulator_;

  static ComplexTransform const kTransformA = ComplexTransform::kNone;
  static ComplexTransform const kTransformB = ComplexTransform::kNone;

  static FloatRoundStyle const Round = cutlass::FloatRoundStyle::round_to_nearest;

  // number of return elements in a global access
  static int const kElementsPerAccess = kElementsPerAccess_;

  using FragmentA = Array<ElementA, kElementsPerAccess>;
  using FragmentB = Array<ElementB, kElementsPerAccess>;

  static int const kBanks = 32;
  static int const kBitsPerSegment = 4;
  static int const kSegmentsPerQuant = sizeof_bits<ElementQ>::value / kBitsPerSegment;
  static int const kBankGroups = kBanks / kSegmentsPerQuant;
  static int const kThreadCount = (kThreadCount_ <= 0) ? 128 : kThreadCount_;
  static int const kStride = (kStride_ <= 0) ? (kElementsPerAccess * kThreadCount * kSegmentsPerQuant) : kStride_;
  static int const kThreadsPerRow = 1 << kBitsPerSegment;
  static int const kQuantsPerAccess =
    (kStride < kElementsPerAccess * kThreadsPerRow * kSegmentsPerQuant) ?
    (kStride / (kThreadsPerRow * kSegmentsPerQuant)) :
    kElementsPerAccess;
  static int const kBlockWidth = kThreadsPerRow * kSegmentsPerQuant * kQuantsPerAccess;

  using FragmentQ = Array<ElementQ, kQuantsPerAccess>;
  using FragmentL = Array<ElementA, kThreadsPerRow>;
  using FragmentCompute = Array<ElementAccumulator, kThreadsPerRow>;

  //
  // Structures
  //

  /// Argument structure
  struct Arguments {
    MatrixCoord     problem_size;
    int32_t         batch_count;

    TensorRefA      ref_A;

    ElementA const *ptr_L;
    ElementB const *ptr_B;
    ElementC const *ptr_C;
    ElementC       *ptr_D;

    int64_t         batch_stride_B;
    int64_t         batch_stride_C;
    int64_t         batch_stride_D;

    //
    // Methods
    //

    Arguments(): batch_count(0) { }

    Arguments(
      MatrixCoord problem_size,
      int32_t     batch_count,
      TensorRefA  ref_A,
      void const *ptr_L,
      void const *ptr_B,
      void const *ptr_C,
      void       *ptr_D,
      int64_t     batch_stride_B,
      int64_t     batch_stride_C,
      int64_t     batch_stride_D
    ):
      problem_size(problem_size),
      batch_count(batch_count),
      ref_A(ref_A),
      ptr_L(static_cast<ElementA const *>(ptr_L)),
      ptr_B(static_cast<ElementB const *>(ptr_B)),
      ptr_C(static_cast<ElementC const *>(ptr_C)),
      ptr_D(static_cast<ElementC       *>(ptr_D)),
      batch_stride_B(batch_stride_B),
      batch_stride_C(batch_stride_C),
      batch_stride_D(batch_stride_D)
    { }

    Status update(Arguments const &args) {
      problem_size = args.problem_size;
      batch_count = args.batch_count;
      ref_A = ref_A;
      ptr_L = ptr_L;
      ptr_B = args.ptr_B;
      ptr_C = args.ptr_C;
      ptr_D = args.ptr_D;
      batch_stride_B = args.batch_stride_B;
      batch_stride_C = args.batch_stride_C;
      batch_stride_D = args.batch_stride_D;

      return Status::kSuccess;
    }
  };

  using Params = Arguments;

  /// Shared memory storage structure
  union SharedStorage {
    AlignedArray<ElementB, kStride> activations;
  };

public:

  //
  // Methods
  //

  CUTLASS_DEVICE
  SqueezeLLMGemv() { }

  /// Determines whether kernel satisfies alignment
  static Status can_implement(cutlass::MatrixCoord const &problem_size) {
    if (problem_size.column() % kElementsPerAccess != 0) {
      return Status::kErrorMisalignedOperand;
    }
    return Status::kSuccess;
  }

  static Status can_implement(Arguments const &args) {
    return can_implement(args.problem_size);
  }

  /// Executes one GEMV
  CUTLASS_DEVICE
  void operator()(Params const &params, SharedStorage &shared_storage) {

    int lane_id = threadIdx.x & (kThreadsPerRow - 1);
    int warp_offset = threadIdx.x & ~(kThreadsPerRow - 1);
    // int warp_id = threadIdx.x / kThreadsPerRow;
    int sync_offset = warp_offset & 31;

    int idx_row_m = blockIdx.y * blockDim.x;
    // problem_size (row = m, column = k)
    // matrix A (batch, m, k)
    // vector B (batch, 1, k)
    // vector C (batch, m, 1)
    // vector D (batch, m, 1)

    // move quant matrix pointer
    int const *ptr_A = params.ref_A.data();
    ptr_A += (idx_row_m + warp_offset) * (params.problem_size.column() / kSegmentsPerQuant);
    ptr_A += blockIdx.x * kStride / kSegmentsPerQuant + lane_id * kQuantsPerAccess;
    // ptr_A += (blockIdx.y * gridDim.x + blockIdx.x) * (kThreadCount * kStride / kSegmentsPerQuant);
    // ptr_A += warp_offset * (kStride / kSegmentsPerQuant) + lane_id * kQuantsPerAccess;

    // move in the m dimension
    ElementA const *ptr_L = params.ptr_L + (idx_row_m + warp_offset) * kThreadsPerRow;
    ElementB const *ptr_B = params.ptr_B;
    ElementC const *ptr_C = params.ptr_C + idx_row_m + threadIdx.x;
    ElementC *ptr_D = params.ptr_D + idx_row_m + threadIdx.x;

    // move in the k dimension
    ptr_B += blockIdx.x * kStride;

    // load LUT
    FragmentL fragL;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kThreadsPerRow; i++) {
      fragL[i] = ptr_L[i * kThreadsPerRow + lane_id];
    }

    // Loop over batch indices
    for (int batch_idx = blockIdx.z; batch_idx < params.batch_count; batch_idx += gridDim.z) {

      // load activations
      CUTLASS_PRAGMA_UNROLL
      for (int offset = threadIdx.x * kElementsPerAccess; offset < kStride; offset += kThreadCount * kElementsPerAccess) {
        reinterpret_cast<FragmentB*>(&shared_storage.activations[offset])[0] =
          reinterpret_cast<const FragmentB*>(&ptr_B[offset])[0];
      }
      __syncthreads();

      FragmentCompute accum;
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kThreadsPerRow; i++) {
        accum.at(i) = 0.0f;
      }

      FragmentQ fragQ;

      CUTLASS_PRAGMA_UNROLL
      for (int unroll_row_m = 0; unroll_row_m < kThreadsPerRow; unroll_row_m++) {

        CUTLASS_PRAGMA_UNROLL
        for (int unroll_col_k = 0; unroll_col_k < kStride; unroll_col_k += kBlockWidth) {

          fragQ = reinterpret_cast<const FragmentQ*>(
            &ptr_A[(unroll_row_m * params.problem_size.column() + unroll_col_k) / kSegmentsPerQuant]
            // &ptr_A[(unroll_row_m * kStride + unroll_col_k) / kSegmentsPerQuant]
          )[0];

          CUTLASS_PRAGMA_UNROLL
          for (int e = 0; e < kQuantsPerAccess; e++) {
            CUTLASS_PRAGMA_UNROLL
            for (int s = 0; s < kSegmentsPerQuant; s++) {
              int k = (s + threadIdx.x / kBankGroups) % kSegmentsPerQuant;
              int idx = (fragQ.at(e) >> (k * kBitsPerSegment)) & (kThreadsPerRow - 1);
              ElementA val = __shfl_sync(0xFFFFFFFF, fragL.at(unroll_row_m), sync_offset + idx);
              ElementB act = shared_storage.activations[unroll_col_k + (lane_id * kQuantsPerAccess + e) * kSegmentsPerQuant + k];
              accum.at(unroll_row_m) += val * act;
            }
          }

        }

      }

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kThreadsPerRow; i++) {
        CUTLASS_PRAGMA_UNROLL
        for (int mask = (kThreadsPerRow >> 1); mask > 0; mask >>= 1) {
          accum.at(i) += __shfl_xor_sync(0xFFFFFFFF, accum.at(i), mask, 32);
        }
      }

      atomicAdd(ptr_D, accum.at(lane_id));

      // move in the batch dimension
      ptr_B += params.batch_stride_B;
      ptr_C += params.batch_stride_C;
      ptr_D += params.batch_stride_D;

    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
