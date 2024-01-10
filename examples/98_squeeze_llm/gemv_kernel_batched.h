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

    ElementQ const *gmem_quant_ptr;
    ElementA const *gmem_sparse_ptr;
    int const *gmem_row_ptr;
    int const *gmem_col_ptr;
    ElementA *smem_ptr;
    ElementA const *lut_ptr;
    FragmentQ frag;
    FragmentL look_up_table;
    ElementA sparse_val;
    int sparse_col;
    int sparse_nnz;
    int lane_id;
    int warp_offset;
    int sync_offset;

    CUTLASS_DEVICE
    WeightIterator(
      ElementQ const *gmem_quant_ptr_,
      ElementA const *gmem_sparse_ptr_,
      int const *gmem_row_ptr_,
      int const *gmem_col_ptr_,
      ElementA const *lut_ptr_,
      ElementA *smem_ptr_
    ):
      gmem_quant_ptr(gmem_quant_ptr_),
      gmem_sparse_ptr(gmem_sparse_ptr_),
      gmem_row_ptr(gmem_row_ptr_),
      gmem_col_ptr(gmem_col_ptr_),
      smem_ptr(smem_ptr_),
      lut_ptr(lut_ptr_)
    {
      lane_id = threadIdx.x & (kThreadsPerRow - 1);
      warp_offset = threadIdx.x & ~(kThreadsPerRow - 1);
      sync_offset = warp_offset & 31;

      gmem_quant_ptr += warp_offset * kQuantBlockWidth + lane_id * kQuantsPerAccess;
      smem_ptr += warp_offset * SharedStorage::kStrideRow;
      lut_ptr += warp_offset * kThreadsPerRow;

      int sparse_start = gmem_row_ptr[threadIdx.x];
      int sparse_end = gmem_row_ptr[threadIdx.x + 1];
      sparse_nnz = sparse_end - sparse_start;
      gmem_sparse_ptr += sparse_start;
      gmem_col_ptr += sparse_start;
      next_sparse_val();

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
      arch::global_load<FragmentQ, sizeof(FragmentQ), arch::CacheOperation::LastUse>(frag, gmem_quant_ptr, true);
      gmem_quant_ptr += kQuantsPerAccess * kThreadsPerRow;
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
    void next_sparse_val() {
      if (sparse_nnz == 0) {
        sparse_col = -1;
      } else {
        sparse_val = gmem_sparse_ptr[0];
        sparse_col = gmem_col_ptr[0];
        gmem_sparse_ptr++;
        gmem_col_ptr++;
        sparse_nnz--;
      }
    }

    CUTLASS_DEVICE
    void next_block(int& unroll_col_k, int &advance, int &index, int &offset) {
      CUTLASS_PRAGMA_UNROLL
      for (int k = 0; k < kThreadsPerRow; k++) {
        smem_ptr[k * SharedStorage::kStrideRow + lane_id * SharedStorage::kStrideCol] =
          __shfl_sync(0xFFFFFFFF, look_up_table.at(k), sync_offset + next_segment(advance, index, offset));
      }
      while ((sparse_col & ~(kThreadsPerRow - 1)) == unroll_col_k) {
        smem_ptr[lane_id * SharedStorage::kStrideRow + (sparse_col & (kThreadsPerRow - 1))] += sparse_val;
        next_sparse_val();
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

    int block_idx = blockIdx.y * gridDim.x + blockIdx.x;

    // move quant matrix pointer
    ElementQ const *ptr_A = params.ref_A.data();
    ptr_A += block_idx * (kThreadCount * kQuantBlockWidth);

    // move sparse matrix pointer
    ElementA const *ptr_S = params.ptr_S;
    int32_t const *ptr_row = params.ptr_row + block_idx * kThreadCount;
    int32_t const *ptr_col = params.ptr_col;

    // move in the m dimension
    ElementA const *ptr_L = params.ptr_L + idx_row_m * kThreadsPerRow;
    ElementB const *ptr_B = params.ptr_B;
    // ElementC const *ptr_C = params.ptr_C + idx_row_m + threadIdx.x;
    ElementC *ptr_D = params.ptr_D + idx_row_m + threadIdx.x;

    // move in the k dimension
    ptr_B += blockIdx.x * kStride;

    WeightIterator iterator_A(ptr_A, ptr_S, ptr_row, ptr_col, ptr_L, &shared_storage.weight[0]);

    FragmentCompute accum;
    CUTLASS_PRAGMA_UNROLL
    for (int batch_idx = 0; batch_idx < kBatchSize; batch_idx++) {
      accum.at(batch_idx) = 0.0f;
    }

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

      iterator_A.next_block(unroll_col_k, advance, index, offset);

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
