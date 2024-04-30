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
  // typename EpilogueOutputOp_,
  int kElementsPerAccess_ = 1,            ///< Number of elements involved in a global access.
  int kThreadCount_ = 0,                  ///< Number of threads in the thread block.
                                          ///  It will be calculated automatically if set to 0.
  int kThreadsPerCol_ = 0                 ///< Number of threads in the k dimension.
                                          ///  It will be calculated automatically if set to 0.
>
struct PitGemv;

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Specializations
//
/////////////////////////////////////////////////////////////////////////////////////////////////

template <
    typename ElementA_,
    typename ElementB_,
    typename ElementC_,
    typename ElementAccumulator_,
    // typename EpilogueOutputOp_,
    int kElementsPerAccess_,
    int kThreadCount_,
    int kThreadsPerCol_ 
>
struct PitGemv <
    ElementA_,            
    layout::RowMajor,
    ElementB_,            
    ElementC_,
    ElementAccumulator_,
    // EpilogueOutputOp_,
    kElementsPerAccess_,
    kThreadCount_,
    kThreadsPerCol_
>{
public:

  using ElementA = ElementA_;
  using LayoutA = layout::RowMajor;
  using TensorRefA = TensorRef<ElementA, LayoutA>;

  using ElementB = ElementB_;
  using ElementC = ElementC_;

  using ElementAccumulator = ElementAccumulator_;
  // using EpilogueOutputOp = EpilogueOutputOp_;

  static ComplexTransform const kTransformA = ComplexTransform::kNone;
  static ComplexTransform const kTransformB = ComplexTransform::kNone;

  static FloatRoundStyle const Round = cutlass::FloatRoundStyle::round_to_nearest;

  // number of return elements in a global access
  static int const kElementsPerAccess = kElementsPerAccess_;
  
  // using FragmentIndex = Array<int32_t, 8>;
  using FragmentA = Array<ElementA, kElementsPerAccess>;
  using FragmentCompute = Array<ElementAccumulator, kElementsPerAccess>;
  using FragmentD = Array<ElementC, kElementsPerAccess>;

  // thread block shape (kThreadsPerCol, kThreadCount / kThreadsPerCol, 1)
  static int const kThreadCount = (kThreadCount_ <= 0) ? 128 : kThreadCount_;
  static int const kThreadsPerCol = (kThreadsPerCol_ <= 0) ? 32 : kThreadsPerCol_;
  static int const kThreadsPerRow = kThreadCount / kThreadsPerCol;
  static int const kBlockWidth = kThreadsPerRow * kElementsPerAccess;
  static int const kWarpSize = 32;

  //
  // Structures
  //

  /// Argument structure
  struct Arguments {
    MatrixCoord     problem_size;
    int32_t         batch_count;
    // typename EpilogueOutputOp::Params output_op;

    TensorRefA      ref_A;

    ElementB const *ptr_B;
    ElementC const *ptr_C;
    ElementC       *ptr_D;

    int32_t         rows_count;
    int16_t  const *rows_index;

    int64_t         batch_stride_A;
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
      // typename EpilogueOutputOp::Params output_op,
      TensorRefA  ref_A,
      void const *ptr_B,
      void const *ptr_C,
      void       *ptr_D,
      int32_t     rows_count,
      void const *rows_index,
      int64_t     batch_stride_A,
      int64_t     batch_stride_B,
      int64_t     batch_stride_C,
      int64_t     batch_stride_D
    ):
      problem_size(problem_size),
      batch_count(batch_count),
      // output_op(output_op),
      ref_A(ref_A),
      ptr_B(static_cast<ElementB const *>(ptr_B)),
      ptr_C(static_cast<ElementC const *>(ptr_C)),
      ptr_D(static_cast<ElementC       *>(ptr_D)),
      rows_count(rows_count),
      rows_index(static_cast<int16_t const *>(rows_index)),
      batch_stride_A(batch_stride_A),
      batch_stride_B(batch_stride_B),
      batch_stride_C(batch_stride_C),
      batch_stride_D(batch_stride_D)
    { }

    Arguments(
      MatrixCoord problem_size,
      // typename EpilogueOutputOp::Params output_op,
      TensorRefA  ref_A,
      void const *ptr_B,
      void const *ptr_C,
      void       *ptr_D,
      int32_t     rows_count,
      void const *rows_index
    ):
      Arguments(
        problem_size,
        1,
        // output_op,
        ref_A,
        ptr_B,
        ptr_C,
        ptr_D,
        rows_count,
        rows_index,
        1,
        1,
        1,
        1)
    { }

    Status update(Arguments const &args) {
      problem_size = args.problem_size;
      batch_count = args.batch_count;
      // output_op = args.output_op;
      ref_A = ref_A;
      ptr_B = args.ptr_B;
      ptr_C = args.ptr_C;
      ptr_D = args.ptr_D;
      rows_count = args.rows_count;
      rows_index = args.rows_index;
      batch_stride_A = args.batch_stride_A;
      batch_stride_B = args.batch_stride_B;
      batch_stride_C = args.batch_stride_C;
      batch_stride_D = args.batch_stride_D;

      return Status::kSuccess;
    }
  };

  using Params = Arguments;

  /// Shared memory storage structure
  union SharedStorage {

  };

public:

  //
  // Methods
  //

  CUTLASS_DEVICE
  PitGemv() {}

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
    int const idx_row_k = threadIdx.x;
    int const idx_col_n = blockIdx.x * kBlockWidth + threadIdx.y * kElementsPerAccess;
    // int const warp_id = (threadIdx.y * blockDim.x + threadIdx.x) / kWarpSize;
    // int const idx_row_k = (warp_id / blockDim.y) * kWarpSize + threadIdx.x % kWarpSize;
    // int const idx_col_n = blockIdx.x * kBlockWidth + (warp_id % blockDim.y) * kElementsPerAccess;
    int const lane_id = threadIdx.x % kWarpSize;

    // Loop over batch indices
    for (int batch_idx = blockIdx.z; batch_idx < params.batch_count; batch_idx += gridDim.z) {

      if (idx_col_n < params.problem_size.column()) {
        // problem_size (row = K = 13824, column = N = 5120)
        // matrix A (batch, K, N)
        // vector B (batch, K)
        // vector C (batch, N)
        // vector D (batch, N)

        // move in the batch dimension
        ElementA const *ptr_A = params.ref_A.data() + batch_idx * params.batch_stride_A;
        ElementB const *ptr_B = params.ptr_B + batch_idx * params.batch_stride_B;

        // ElementC const *ptr_C = params.ptr_C + batch_idx * params.batch_stride_C;
        // ElementC *ptr_D = params.ptr_D + batch_idx * params.batch_stride_D;
        half *ptr_D = reinterpret_cast<half *>(&params.ptr_D[batch_idx * params.batch_stride_D]);

        int16_t const *rows_index = reinterpret_cast<int16_t const *>(&params.rows_index[0]);

        // move in the K dimension
        // ptr_A += idx_row_k * params.problem_size.column();
        // ptr_B += idx_row_k;

        // move in the N dimension
        ptr_A += idx_col_n;
        // ptr_C += idx_col_n;
        ptr_D += idx_col_n;

        NumericArrayConverter<ElementAccumulator, ElementA, kElementsPerAccess, Round> srcA_converter;
        NumericArrayConverter<ElementC, ElementAccumulator, kElementsPerAccess, Round> dstD_converter;

        FragmentA fragA;
        FragmentCompute accum;

        CUTLASS_PRAGMA_UNROLL
        for (int e = 0; e < kElementsPerAccess; e++) {
          accum.at(e) = 0.0f;
        }

        int unroll_col_k = idx_row_k;

        for (; unroll_col_k < params.rows_count; unroll_col_k += kThreadsPerCol) {
          int k = rows_index[unroll_col_k];

          // fetch from matrix A
          arch::global_load<FragmentA,
                            sizeof(FragmentA),
                            arch::CacheOperation::LastUse>(fragA, (ptr_A + k * params.problem_size.column()), true);

          // fetch from vector B
          ElementB b = *(ptr_B + k);

          FragmentCompute fragA_Compute = srcA_converter(fragA);

          // Math
          CUTLASS_PRAGMA_UNROLL
          for (int e = 0; e < kElementsPerAccess; e++) {
            accum.at(e) += fragA_Compute.at(e) * b;
          }
        }

        // EpilogueOutputOp output_op(params.output_op);
        // typename EpilogueOutputOp::FragmentOutput source_fragment;

        // prefetch from source matrix C
        // if (output_op.is_source_needed()) {         
        //   source_fragment[0] = *(ptr_C);
        // }

        // typename EpilogueOutputOp::FragmentAccumulator accum_fragment;
        // typename EpilogueOutputOp::FragmentOutput output_fragment;

        CUTLASS_PRAGMA_UNROLL
        for (int e = 0; e < kElementsPerAccess; e++) {
          CUTLASS_PRAGMA_UNROLL
          for (int mask = (kWarpSize >> 1); mask > 0; mask >>= 1) {
            accum.at(e) += __shfl_xor_sync(0xFFFFFFFF, accum.at(e), mask, kWarpSize);
          }
        }

        // if (idx_row_k == 0) {
        //   FragmentD fragD = dstD_converter(accum);
        //   // accum_fragment[0] = accum;

        //   // if (output_op.is_source_needed()) {
        //   //   output_fragment = output_op(accum_fragment, source_fragment);
        //   // }
        //   // else {
        //   //   output_fragment = output_op(accum_fragment);
        //   // }

        //   // *ptr_D = output_fragment[0];
        //   arch::global_store<FragmentD, sizeof(FragmentD)>(fragD, ptr_D, true);
        // }

        if (lane_id < kElementsPerAccess) {
          atomicAdd(ptr_D + lane_id, __float2half(accum.at(lane_id)));
        }
      }
    }
  }
};

} // namespace kernel
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
