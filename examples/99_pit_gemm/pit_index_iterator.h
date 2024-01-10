#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/arch/memory.h"
#include "cutlass/layout/layout.h"

namespace cutlass {
namespace transform {
namespace threadblock {

namespace pit {

// SmemSize >= kBlocksPerTile = (128 / 8) or (128 / 1)
constexpr unsigned int SmemPow = 9;
constexpr unsigned int SmemStages = 2;
constexpr unsigned int SmemSize = 1 << SmemPow;
constexpr unsigned int SmemMask = (SmemSize * SmemStages - 1);
constexpr unsigned int AccessNum = 8;

class SharedStorage {
  public:
    Array<int16_t, SmemSize*SmemStages> array;
};

template <typename ThreadBlockShape_, int PitBlockShape_>
class IndexIterator {

  public:
    using Layout = layout::PitchLinear;
    using BytePointer = char *;
    using TensorCoord = Layout::TensorCoord;
    static int const PitBlockShape = PitBlockShape_;

    static int const kBlocksPerTile = ThreadBlockShape_::kContiguous / PitBlockShape_;
    static int const kTilesPerLoad = SmemSize / kBlocksPerTile;

  private:
    const int16_t *gmem_ptr_;
    int16_t *smem_ptr_;
    int kBlocksPerRow; // number of PIT blocks in a row
    int tile_offset_; // along the contiguous axis
    int block_offset_; // along the contiguous axis
    int smem_stage_;

    CUTLASS_DEVICE
    void load_indices_sync_() {
      CUTLASS_PRAGMA_UNROLL
      for (int i = threadIdx.x * 8; i < SmemSize; i += blockDim.x * 8) {
        int load_offset = (block_offset_ + i < kBlocksPerRow) ? block_offset_ + i : kBlocksPerRow - 8;
        using AccessType  = Array<int16_t, 8>;
        *reinterpret_cast<AccessType *>(&smem_ptr_[smem_stage_ * SmemSize + i]) =
          *reinterpret_cast<AccessType const *>(&gmem_ptr_[load_offset]);
      }
      __syncthreads();
      smem_stage_ ^= 1;
      tile_offset_ += kTilesPerLoad;
      block_offset_ += SmemSize;
    }

    CUTLASS_DEVICE
    void load_indices_async_() {
      CUTLASS_PRAGMA_UNROLL
      for (int i = threadIdx.x * AccessNum; i < SmemSize; i += blockDim.x * AccessNum) {
        int load_offset = (block_offset_ + i < kBlocksPerRow) ? block_offset_ + i : kBlocksPerRow - AccessNum;
        cutlass::arch::cp_async<sizeof(int16_t) * AccessNum>(&smem_ptr_[smem_stage_ * SmemSize + i], &gmem_ptr_[load_offset]);
      }
      cutlass::arch::cp_async_fence();
      cutlass::arch::cp_async_wait<SmemStages - 2>();
      smem_stage_ ^= 1;
      tile_offset_ += kTilesPerLoad;
      block_offset_ += SmemSize;
    }

  public:
    CUTLASS_DEVICE
    IndexIterator(
      SharedStorage& shared_storage_base,
      const int16_t* gmem_ptr,
      int num_rows, // of the sparse tensor
      int tb_offset, // of the sparse tensor
      int thread_id)
      : smem_ptr_(reinterpret_cast<int16_t*>(&shared_storage_base.array)),
        gmem_ptr_(const_cast<int16_t*>(gmem_ptr)) {

      kBlocksPerRow = (num_rows + PitBlockShape - 1) / PitBlockShape;
      gmem_ptr_ += tb_offset * kBlocksPerRow;
      tile_offset_ = 0;
      block_offset_ = 0;
      smem_stage_ = 0;
      load_indices_sync_();
    }

    CUTLASS_DEVICE
    void load_indices(int target_tile_offset) {
      if (tile_offset_ <= target_tile_offset && block_offset_ < kBlocksPerRow) {
        load_indices_sync_();
      }
    }

    CUTLASS_DEVICE
    int get_advance_offset(int advance) {
      int x = advance / PitBlockShape;
      int block_idx = smem_ptr_[x & SmemMask];
      // int block_idx = gmem_ptr_[x];
      return (block_idx - x) * PitBlockShape;
    }
};

}
}
}
}
