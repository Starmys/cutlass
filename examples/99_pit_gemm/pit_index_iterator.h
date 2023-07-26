#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/arch/memory.h"
#include "cutlass/layout/layout.h"

namespace cutlass {
namespace transform {
namespace threadblock {

namespace pit {

// SmemSize >= blockDim.x * (sizeof(AccessType) / sizeof(Index)) = 128 * 8
// SmemSize >= ThreadBlockShape / PitBlockShape = (128 * 32) / (1 * 8)
constexpr unsigned int SmemPow = 10;
constexpr unsigned int SmemStages = 2;
constexpr unsigned int SmemSize = 1 << SmemPow;
constexpr unsigned int SmemMask = (SmemSize * SmemStages - 1);

class SharedStorage {
  public:
    Array<int, SmemSize*SmemStages> array;
};

template <typename ThreadBlockShape_, typename PitBlockShape_>
class IndexIterator {

  public:
    using Layout = layout::PitchLinear;
    using BytePointer = char *;
    using AccessType = AlignedArray<int, 8>;
    using TensorCoord = Layout::TensorCoord;
    using ThreadBlockShape = ThreadBlockShape_;
    using PitBlockShape = PitBlockShape_;

    static int const kBlocksPerTileContiguous = ThreadBlockShape_::kContiguous / PitBlockShape_::kContiguous;
    static int const kBlocksPerTileStrided = ThreadBlockShape_::kStrided / PitBlockShape_::kStrided;
    static int const kTilesPerLoad = SmemSize / (kBlocksPerTileContiguous * kBlocksPerTileStrided);
    static int const kBlocksPerLoadContiguous = kTilesPerLoad * kBlocksPerTileContiguous;

  private:
    const int *gmem_ptr_;
    const BytePointer zero_ptr_; // point to an array of zeros in the global memory
    int *smem_ptr_;
    AccessType frag;
    TensorCoord extent_;
    int blocks_per_row; // number of PIT blocks in a row
    int tile_offset_; // along the contiguous axis
    int block_offset_; // along the contiguous axis
    int smem_stage_;

  public:
    CUTLASS_DEVICE
    IndexIterator(
      SharedStorage& shared_storage_base,
      const int* gmem_ptr,
      const BytePointer zero_ptr,
      TensorCoord extent, // of the sparse tensor
      TensorCoord const &threadblock_offset, // of the sparse tensor
      int thread_id)
      : smem_ptr_(reinterpret_cast<int*>(&shared_storage_base.array)),
        gmem_ptr_(const_cast<int*>(gmem_ptr)),
        zero_ptr_(zero_ptr),
        extent_(extent) {

      blocks_per_row = (extent_.contiguous() + PitBlockShape::kContiguous - 1) / PitBlockShape::kContiguous;
      gmem_ptr += threadblock_offset.strided() / PitBlockShape::kStrided * blocks_per_row;
      tile_offset_ = 0;
      block_offset_ = 0;
      smem_stage_ = 0;
      load_indices(0);
      __syncthreads();
    }

    CUTLASS_DEVICE
    void load_indices(int target_tile_offset) {
      if (tile_offset_ <= target_tile_offset && block_offset_ < blocks_per_row) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = threadIdx.x * AccessType::kElements; i < SmemSize; i += blockDim.x * AccessType::kElements) {
          int y = i / kBlocksPerLoadContiguous;
          int x = block_offset_ + i % kBlocksPerLoadContiguous;
          if (x < blocks_per_row) {
            AccessType const *access_ptr = reinterpret_cast<AccessType const *>(&gmem_ptr_[y * blocks_per_row + x]);
            arch::global_load<AccessType, sizeof(AccessType), arch::CacheOperation::LastUse>(frag, access_ptr, true);
            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < AccessType::kElements; j++) {
              smem_ptr_[smem_stage_ * SmemSize + i + j] = frag.at(j) - 1;
            }
          }
        }
        smem_stage_ ^= 1;
        tile_offset_ += kTilesPerLoad;
        block_offset_ += kBlocksPerLoadContiguous;
      }
      // if (threadIdx.x == 0) {
      //   std::printf(
      //     "[bid = %d] target_tile_offset = %d, tile_offset_ = %d, block_offset_ = %d, smem[0] = %d\n",
      //     int(blockIdx.x),
      //     int(target_tile_offset),
      //     int(tile_offset_),
      //     int(block_offset_),
      //     int(smem_ptr_[smem_stage_ * SmemSize])
      //   );
      // }
      // if (blockIdx.x == 0) {
      //   TensorCoord thread_offset = ThreadMap::initial_offset(thread_id);
      //   std::printf(
      //     "[tid = (%d)] thread_offset = (%d, %d)\n",
      //     int(threadIdx.x),
      //     int(thread_offset.strided()),
      //     int(thread_offset.contiguous())
      //   );
      // }
    }

    CUTLASS_DEVICE
    int get_real_offset(const int& offset) {
      int y = ((offset / extent_.contiguous()) % ThreadBlockShape::kStrided) / PitBlockShape::kStrided;
      int x = (offset % extent_.contiguous()) / PitBlockShape::kContiguous;
      // int smem_offset = (x / kBlocksPerLoadContiguous * kBlocksPerTileStrided + y) * kBlocksPerLoadContiguous
      //                 + x % kBlocksPerLoadContiguous;
      // int block_idx = smem_ptr_[smem_offset & SmemMask];
      int block_idx = gmem_ptr_[y * blocks_per_row + x] - 1;
      if (blockIdx.x == 1) {
        std::printf(
          "[bid = %d, tid = %d] y = %d, x = %d -> %d\n",
          int(blockIdx.x),
          int(threadIdx.x),
          int(offset / extent_.contiguous()),
          int(x),
          int(block_idx)
        );
      }
      if (block_idx < 0) {
        return -1;
      } else {
        return offset + (block_idx - x) * PitBlockShape::kContiguous;
      }
    }

    CUTLASS_DEVICE
    BytePointer get_zero_pointer() {
      return zero_ptr_;
    }
};

}
}
}
}
