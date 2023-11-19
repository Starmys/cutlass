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
  int kStride_ = 0,
  int kBatchSize_ = 1
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
    kStride_,
    1
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

  static int const kBitsPerSegment = 4;
  static int const kThreadCount = (kThreadCount_ <= 0) ? 128 : kThreadCount_;
  static int const kStride = (kStride_ <= 0) ? 128 : kStride_;
  static int const kThreadsPerRow = 1 << kBitsPerSegment;
  static int const kWarps = kThreadCount / kThreadsPerRow;
  static int const kQuantBlockWidth = kStride * kBitsPerSegment / sizeof_bits<ElementQ>::value;
  static int const kQuantsPerAccess = 4;
  static int const kResidualQuants = kQuantBlockWidth % kQuantsPerAccess;

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
    ElementA const *ptr_S;
    int32_t  const *ptr_row;
    int32_t  const *ptr_col;
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
      void const *ptr_S,
      void const *ptr_row,
      void const *ptr_col,
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
      ptr_S(static_cast<ElementA const *>(ptr_S)),
      ptr_row(static_cast<int32_t const *>(ptr_row)),
      ptr_col(static_cast<int32_t const *>(ptr_col)),
      ptr_B(static_cast<ElementB const *>(ptr_B)),
      ptr_C(static_cast<ElementC const *>(ptr_C)),
      ptr_D(static_cast<ElementC       *>(ptr_D)),
      batch_stride_B(batch_stride_B),
      batch_stride_C(batch_stride_C),
      batch_stride_D(batch_stride_D)
    {
      assert(batch_count == 1);
    }

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
    AlignedArray<half, kStride> activations;
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
    int sync_offset = warp_offset & 31;

    int idx_row_m = blockIdx.y * blockDim.x;
    // problem_size (row = m, column = k)
    // matrix A (batch, m, k)
    // vector B (batch, 1, k)
    // vector C (batch, m, 1)
    // vector D (batch, m, 1)

    int block_idx = blockIdx.y * gridDim.x + blockIdx.x;

    // move quant matrix pointer
    ElementQ const *ptr_A = params.ref_A.data();
    ptr_A += block_idx * (kThreadCount * kQuantBlockWidth);
    // ptr_A += (warp_offset + lane_id) * kQuantBlockWidth;
    ptr_A += warp_offset * kQuantBlockWidth + lane_id * kQuantsPerAccess;

    // move sparse matrix pointer
    int32_t const *ptr_row = params.ptr_row + block_idx * kThreadCount;
    int sparse_start = ptr_row[threadIdx.x];
    int sparse_end = ptr_row[threadIdx.x + 1];
    ElementA const *ptr_S = params.ptr_S;
    int32_t const *ptr_col = params.ptr_col;

    // move in the m dimension
    ElementA const *ptr_L = params.ptr_L + (idx_row_m + warp_offset) * kThreadsPerRow;
    // ElementB const *ptr_B = params.ptr_B;
    half const *ptr_B = reinterpret_cast<const half*>(&params.ptr_B[0]);
    // ElementC const *ptr_C = params.ptr_C + idx_row_m + threadIdx.x;
    // ElementC *ptr_D = params.ptr_D + idx_row_m + threadIdx.x;
    half *ptr_D = reinterpret_cast<half*>(&params.ptr_D[idx_row_m + threadIdx.x]);

    // move in the k dimension
    ptr_B += blockIdx.x * kStride;

    // load LUT
    FragmentL fragL;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kThreadsPerRow; i++) {
      fragL[i] = ptr_L[i * kThreadsPerRow + lane_id];
    }

    // load activations
    CUTLASS_PRAGMA_UNROLL
    for (int offset = threadIdx.x * kElementsPerAccess; offset < kStride; offset += kThreadCount * kElementsPerAccess) {
      reinterpret_cast<FragmentB*>(&shared_storage.activations[offset])[0] =
        reinterpret_cast<const FragmentB*>(&ptr_B[offset])[0];

      if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("# Load [%d][%d] %f\n", lane_id, offset, __half2float(half(shared_storage.activations[offset])));
      }
    }
    __syncthreads();

    FragmentCompute accum;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kThreadsPerRow; i++) {
      accum.at(i) = (ElementAccumulator)(0.0f);
    }

    FragmentQ fragQ;
    int residual;

    CUTLASS_PRAGMA_UNROLL
    for (int quant_k = 0, offset = 0, k = 0; quant_k < kQuantBlockWidth; quant_k += kQuantsPerAccess) {

      arch::global_load<FragmentQ, sizeof(FragmentQ), arch::CacheOperation::LastUse>(
        fragQ,
        // ptr_A + quant_k,
        ptr_A + quant_k * kThreadsPerRow,
        true
      );

      CUTLASS_PRAGMA_UNROLL
      for (int e = 0; e < kQuantsPerAccess; e++) {
        if (offset > 0) {
          int unroll_row_m = k & (kThreadsPerRow - 1);
          int unroll_col_k = k & ~(kThreadsPerRow - 1);
          int idx = ((fragQ.at(e) & ((1 << offset) - 1)) << (kBitsPerSegment - offset)) | residual;
          ElementA val = __shfl_sync(0xFFFFFFFF, fragL.at(unroll_row_m), sync_offset + idx);
          half act = shared_storage.activations[unroll_col_k + lane_id];
          accum.at(unroll_row_m) += val * __half2float(act);
          k++;
        }
        CUTLASS_PRAGMA_UNROLL
        for (int q = 0; q < sizeof_bits<ElementQ>::value / kBitsPerSegment; q++, offset += kBitsPerSegment, k++) {
          int unroll_row_m = k & (kThreadsPerRow - 1);
          int unroll_col_k = k & ~(kThreadsPerRow - 1);
          int idx = (fragQ.at(e) >> offset) & (kThreadsPerRow - 1);
          ElementA val = __shfl_sync(0xFFFFFFFF, fragL.at(unroll_row_m), sync_offset + idx);
          half act = shared_storage.activations[unroll_col_k + lane_id];
          accum.at(unroll_row_m) += val * __half2float(act);

          // if (blockIdx.x == 0 && threadIdx.x == 0) {
          //   printf("# Input [%d][%d] %f\n", lane_id, k, __half2float(act));
          // }
        }
        if (offset != sizeof_bits<ElementQ>::value) {
          residual = fragQ.at(e) >> offset;
        }
        offset %= sizeof_bits<ElementQ>::value;
      }

    }

    CUTLASS_PRAGMA_UNROLL
    for (int unroll_row_m = 0; unroll_row_m < kThreadsPerRow; unroll_row_m++) {
      CUTLASS_PRAGMA_UNROLL
      for (int mask = (kThreadsPerRow >> 1); mask > 0; mask >>= 1) {
        accum.at(unroll_row_m) += __shfl_xor_sync(0xFFFFFFFF, accum.at(unroll_row_m), mask);
      }
    }

    // sparse
    for (int sparse_idx = sparse_start; sparse_idx < sparse_end; sparse_idx++) {
      half act = shared_storage.activations[ptr_col[sparse_idx]];
      accum.at(lane_id) += ptr_S[sparse_idx] * __half2float(act);
    }

    if (blockIdx.x == 0 && threadIdx.x == 0) {
      printf("# Output [%d] %f\n", lane_id, accum.at(lane_id));
    }

    atomicAdd(ptr_D, __float2half(accum.at(lane_id)));

  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
