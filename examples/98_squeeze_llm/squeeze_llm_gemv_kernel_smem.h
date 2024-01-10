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

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Specializations
//
/////////////////////////////////////////////////////////////////////////////////////////////////

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

  using ElementA = ElementA_;
  using LayoutA = layout::RowMajor;
  using TensorRefA = TensorRef<int, LayoutA>;

  using ElementB = ElementB_;
  using ElementC = ElementC_;

  using ElementAccumulator = ElementAccumulator_;

  static ComplexTransform const kTransformA = ComplexTransform::kNone;
  static ComplexTransform const kTransformB = ComplexTransform::kNone;

  static FloatRoundStyle const Round = cutlass::FloatRoundStyle::round_to_nearest;

  // number of return elements in a global access
  static int const kElementsPerAccess = kElementsPerAccess_;

  using FragmentQ = Array<int, kElementsPerAccess>;
  using FragmentA = Array<ElementA, kElementsPerAccess>;
  using FragmentB = Array<ElementB, kElementsPerAccess>;
  using FragmentCompute = Array<ElementAccumulator, kElementsPerAccess>;

  static int const kBitsPerSegment = 4;
  static int const kSegmentsPerQuant = 32 / kBitsPerSegment;
  static int const kThreadCount = (kThreadCount_ <= 0) ? 128 : kThreadCount_;
  static int const kStride = (kStride_ <= 0) ? kElementsPerAccess : kStride_;
  static int const kAccessesPerQuant = 32 / kElementsPerAccess;
  static int const kValsPerSegment = 1 << kBitsPerSegment;
  static int const kBlockWidth = kSegmentsPerQuant * kElementsPerAccess;
  static int const kBlockHeight = kThreadCount;

  using FragmentL = Array<ElementA, kValsPerSegment>;

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

    Arguments(
      MatrixCoord problem_size,
      TensorRefA  ref_A,
      void const *ptr_B,
      void const *ptr_C,
      void       *ptr_D
    ):
      Arguments(
        problem_size,
        1,
        ref_A,
        ptr_B,
        ptr_C,
        ptr_D,
        1,
        1,
        1,
        1)
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
  struct SharedStorage {
    AlignedArray<ElementB, kStride> activations;
    Array<ElementA, (kBlockWidth + 1) * kBlockHeight> weight;
  };

public:

  //
  // Methods
  //

  CUTLASS_DEVICE
  SqueezeLLMGemv() {}

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

    int lane_id = threadIdx.x & (kValsPerSegment - 1);
    int warp_id = threadIdx.x / kValsPerSegment;
    int warp_offset = threadIdx.x & kValsPerSegment;

    int idx_row_m = blockIdx.y * blockDim.x;
    if (idx_row_m + threadIdx.x < params.problem_size.row()) {
      // problem_size (row = m, column = k)
      // matrix A (batch, m, k)
      // vector B (batch, 1, k)
      // vector C (batch, m, 1)
      // vector D (batch, m, 1)

      // move quant matrix pointer
      int const *ptr_A = params.ref_A.data();
      ptr_A += (idx_row_m + )
      ptr_A += (blockIdx.y * gridDim.x + blockIdx.x) * (kBlockHeight * kStride / kSegmentsPerQuant);
      ptr_A += warp_id * (kValsPerSegment * kBlockWidth / kSegmentsPerQuant) + threadIdx.x * kElementsPerAccess;

      // move in the m dimension
      ElementA const *ptr_L = params.ptr_L + (idx_row_m + warp_id * kValsPerSegment) * kValsPerSegment;
      ElementB const *ptr_B = params.ptr_B;
      ElementC const *ptr_C = params.ptr_C + idx_row_m + threadIdx.x;
      ElementC *ptr_D = params.ptr_D + idx_row_m + threadIdx.x;

      // move in the k dimension
      ptr_B += blockIdx.x * kStride;

      // load LUT
      FragmentL fragL;
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kValsPerSegment; i++) {
        fragL[i] = ptr_L[i * kValsPerSegment + lane_id];
      }

      // Loop over batch indices
      for (int batch_idx = blockIdx.z; batch_idx < params.batch_count; batch_idx += gridDim.z) {

        // move in the batch dimension
        ptr_B += batch_idx * params.batch_stride_B;
        ptr_C += batch_idx * params.batch_stride_C;
        ptr_D += batch_idx * params.batch_stride_D;

        // load activations
        CUTLASS_PRAGMA_UNROLL
        for (int offset = threadIdx.x * kElementsPerAccess; offset < kStride; offset += kThreadCount * kElementsPerAccess) {
          reinterpret_cast<FragmentB*>(&shared_storage.activations[offset])[0] =
            reinterpret_cast<const FragmentB*>(&ptr_B[offset])[0];
        }
        __syncthreads();

        NumericArrayConverter<ElementAccumulator, ElementA, kElementsPerAccess, Round> srcA_converter;
        NumericArrayConverter<ElementAccumulator, ElementB, kElementsPerAccess, Round> srcB_converter;

        ElementAccumulator accum = 0.f;

        FragmentB fragB;
        FragmentA fragA;
        FragmentQ fragQ;

        for (int unroll_col_k = 0; unroll_col_k < kStride; unroll_col_k += kBlockWidth) {

          arch::global_load<FragmentQ,
                            sizeof(FragmentQ),
                            arch::CacheOperation::LastUse>(fragQ, ptr_A, true);
          ptr_A += kBlockHeight * kElementsPerAccess;

          // __syncthreads();
          CUTLASS_PRAGMA_UNROLL
          for (int e = 0; e < kElementsPerAccess; e++) {
            CUTLASS_PRAGMA_UNROLL
            for (int s = 0; s < kSegmentsPerQuant; s++) {
              int k = e * kSegmentsPerQuant + s;
              int i = k & (kValsPerSegment - 1);
              int j = (k & kValsPerSegment) + lane_id;
              int idx = (fragQ.at(e) >> (s * kBitsPerSegment)) & (kValsPerSegment - 1);
              ElementA val = __shfl_sync(0xffffffff, fragL.at(i), warp_offset + idx);
              shared_storage.weight[j + (warp_id * kValsPerSegment + i) * (kBlockWidth + 1)] = val;
            }
          }
          // __syncthreads();

          CUTLASS_PRAGMA_UNROLL
          for (int k = 0; k < kBlockWidth; k += kElementsPerAccess) {

            // fetch from weight
            CUTLASS_PRAGMA_UNROLL
            for (int e = 0; e < kElementsPerAccess; e++) {
              fragA.at(e) = shared_storage.weight[(k + e) + threadIdx.x * (kBlockWidth + 1)];
            }

            // fetch from activations
            fragB = reinterpret_cast<FragmentB*>(&shared_storage.activations[unroll_col_k + k])[0];

            FragmentCompute fragB_Compute = srcB_converter(fragB);
            FragmentCompute fragA_Compute = srcA_converter(fragA);

            // Math
            CUTLASS_PRAGMA_UNROLL
            for (int e = 0; e < kElementsPerAccess; e++) {
              accum += fragA_Compute.at(e) * fragB_Compute.at(e);
            }

          }
          // __syncthreads();

        }

        // TODO: calculate the rest of K elements

        atomicAdd(ptr_D, accum);
      }
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
