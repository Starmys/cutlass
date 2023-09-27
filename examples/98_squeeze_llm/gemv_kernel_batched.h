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

#include "gemv_kernel.h"
/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

// GEMV for row-major A matrix
template <
    typename ElementA_,
    typename ElementB_,
    typename ElementC_,
    typename ElementAccumulator_,
    int kElementsPerAccess_,
    int kThreadCount_,
    int kStride_,
    int kBatchSize_
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
    kBatchSize_
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

  static int const kBatchSize = kBatchSize_;
  static int const kBitsPerSegment = 4;
  static int const kThreadCount = (kThreadCount_ <= 0) ? 128 : kThreadCount_;
  static int const kStride = (kStride_ <= 0) ? 128 : kStride_;
  static int const kThreadsPerRow = 1 << kBitsPerSegment;
  static int const kWarps = kThreadCount / kThreadsPerRow;
  static int const kQuantBlockWidth = kStride * kBitsPerSegment / sizeof_bits<ElementQ>::value;
  static int const kQuantsPerAccess = 4;
  static int const kResidualQuants = kQuantBlockWidth % kQuantsPerAccess;
  static int const kActivationStride = ((kThreadCount * kElementsPerAccess) / kBatchSize) & ~(kThreadsPerRow - 1);

  using FragmentQ = Array<ElementQ, kQuantsPerAccess>;
  using FragmentL = Array<ElementA, kThreadsPerRow>;
  using FragmentCompute = Array<ElementAccumulator, kBatchSize>;

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
    {
      assert(batch_count == kBatchSize);
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
  struct SharedStorage {
    static int const kStrideRow = kThreadsPerRow + 1;
    static int const kStrideCol = 1;
    Array<ElementB, kBatchSize * kActivationStride> activations;
    Array<ElementA, kThreadCount * (kThreadsPerRow + 1)> weight;
  };

  /// Quant weight iterator
  struct WeightIterator {

    ElementQ const *gmem_ptr;
    ElementA *smem_ptr;
    ElementA const *lut_ptr;
    FragmentQ frag;
    FragmentL look_up_table;
    int lane_id;
    int warp_offset;
    int sync_offset;

    CUTLASS_DEVICE
    WeightIterator(
      ElementQ const *gmem_ptr_,
      ElementA const *lut_ptr_,
      ElementA *smem_ptr_
    ):
      gmem_ptr(gmem_ptr_),
      smem_ptr(smem_ptr_),
      lut_ptr(lut_ptr_)
    {
      lane_id = threadIdx.x & (kThreadsPerRow - 1);
      warp_offset = threadIdx.x & ~(kThreadsPerRow - 1);
      sync_offset = warp_offset & 31;

      gmem_ptr += warp_offset * kQuantBlockWidth + lane_id * kQuantsPerAccess;
      smem_ptr += warp_offset * SharedStorage::kStrideRow + lane_id * SharedStorage::kStrideCol;
      lut_ptr += warp_offset * kThreadsPerRow;

      load_lut();
    }

    CUTLASS_DEVICE
    void load_lut() {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kThreadsPerRow; i++) {
        look_up_table.at(i) = lut_ptr[i * kThreadsPerRow + lane_id];
      }
    }

    CUTLASS_DEVICE
    void load_weight(int &advance) {
      arch::global_load<FragmentQ, sizeof(FragmentQ), arch::CacheOperation::LastUse>(frag, gmem_ptr, true);
      gmem_ptr += kQuantsPerAccess * kThreadsPerRow;
      advance += kQuantsPerAccess;
    }

    CUTLASS_DEVICE
    ElementQ get_quant(int &advance, int &index) {
      if (advance <= index) {
        load_weight(advance);
      }
      return frag.at(index % kQuantsPerAccess);
    }

    CUTLASS_DEVICE
    ElementQ next_segment(int &advance, int &index, int &offset) {
      ElementQ seg = (get_quant(advance, index) >> offset) & (kThreadsPerRow - 1);
      offset += kBitsPerSegment;
      if (offset > sizeof_bits<ElementQ>::value) {
        index++;
        offset %= sizeof(ElementQ);
        seg |= (get_quant(advance, index) & ((1 << offset) - 1)) << (kBitsPerSegment - offset);
      } else if (offset == sizeof_bits<ElementQ>::value) {
        index++;
        offset = 0;
      }
      return seg;
    }

    CUTLASS_DEVICE
    void next_block(int &advance, int &index, int &offset) {
      CUTLASS_PRAGMA_UNROLL
      for (int k = 0; k < kThreadsPerRow; k++) {
        smem_ptr[k * SharedStorage::kStrideRow] =
          __shfl_sync(0xFFFFFFFF, look_up_table.at(k), sync_offset + next_segment(advance, index, offset));
      }
    }

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

    int idx_row_m = blockIdx.y * blockDim.x;
    // problem_size (row = m, column = k)
    // matrix A (batch, m, k)
    // vector B (batch, 1, k)
    // vector C (batch, m, 1)
    // vector D (batch, m, 1)

    // move quant matrix pointer
    ElementQ const *ptr_A = params.ref_A.data();
    ptr_A += (blockIdx.y * gridDim.x + blockIdx.x) * (kThreadCount * kQuantBlockWidth);

    // move in the m dimension
    ElementA const *ptr_L = params.ptr_L + idx_row_m * kThreadsPerRow;
    ElementB const *ptr_B = params.ptr_B;
    // ElementC const *ptr_C = params.ptr_C + idx_row_m + threadIdx.x;
    ElementC *ptr_D = params.ptr_D + idx_row_m + threadIdx.x;

    // move in the k dimension
    ptr_B += blockIdx.x * kStride;

    WeightIterator iterator_A(ptr_A, ptr_L, &shared_storage.weight[0]);

    FragmentCompute accum;
    CUTLASS_PRAGMA_UNROLL
    for (int batch_idx = 0; batch_idx < kBatchSize; batch_idx++) {
      accum.at(batch_idx) = 0.0f;
    }

    // load activations
    // for (int batch_idx = 0; batch_idx < params.batch_count; batch_idx++) {
    //   CUTLASS_PRAGMA_UNROLL
    //   for (int offset = threadIdx.x * kElementsPerAccess; offset < kStride; offset += kThreadCount * kElementsPerAccess) {
    //     reinterpret_cast<FragmentB*>(&shared_storage.activations[batch_idx * kStride + offset])[0] =
    //       reinterpret_cast<const FragmentB*>(&ptr_B[batch_idx * params.batch_stride_B + offset])[0];
    //   }
    // }
    // for (int offset = threadIdx.x * kElementsPerAccess; offset < kStride; offset += kThreadCount * kElementsPerAccess) {
    //   reinterpret_cast<FragmentB*>(&shared_storage.activations[offset])[0] =
    //     reinterpret_cast<const FragmentB*>(&ptr_B[offset])[0];
    // }
    // __syncthreads();

    CUTLASS_PRAGMA_UNROLL
    for (int unroll_col_k = 0, advance = 0, index = 0, offset = 0; unroll_col_k < kStride; unroll_col_k += kThreadsPerRow) {

      if (unroll_col_k % kActivationStride == 0) {
        if (unroll_col_k > 0) __syncthreads();
        CUTLASS_PRAGMA_UNROLL
        for (
          int k = threadIdx.x * kElementsPerAccess;
          (k < kBatchSize * kActivationStride) && (unroll_col_k + k % kActivationStride < kStride);
          k += kThreadCount * kElementsPerAccess
        ) {
          int batch_idx = k / kActivationStride;
          int offset = k % kActivationStride;
          reinterpret_cast<FragmentB*>(&shared_storage.activations[k])[0] =
            reinterpret_cast<const FragmentB*>(&ptr_B[batch_idx * params.batch_stride_B + unroll_col_k + offset])[0];
        }
        __syncthreads();
      }

      iterator_A.next_block(advance, index, offset);

      CUTLASS_PRAGMA_UNROLL
      for (int batch_idx = 0; batch_idx < kBatchSize; batch_idx++) {
        CUTLASS_PRAGMA_UNROLL
        for (int e = 0; e < kThreadsPerRow; e++) {
          accum.at(batch_idx) += shared_storage.weight[threadIdx.x * SharedStorage::kStrideRow + e * SharedStorage::kStrideCol] *
            shared_storage.activations[batch_idx * kActivationStride + unroll_col_k % kActivationStride + e];
        }
      }

    }

    CUTLASS_PRAGMA_UNROLL
    for (int batch_idx = 0; batch_idx < kBatchSize; batch_idx++) {
      atomicAdd(ptr_D, accum.at(batch_idx));
      ptr_D += params.batch_stride_D;
    }

  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
