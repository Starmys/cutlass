#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/arch/memory.h"
#include "cutlass/layout/layout.h"

namespace cutlass {
namespace transform {
namespace threadblock {

namespace pit {

// SmemSize >= kBlocksPerTiles = (128 / 8) or (128 / 1)
constexpr unsigned int SmemPow = 8;
constexpr unsigned int SmemStages = 2;
constexpr unsigned int SmemSize = 1 << SmemPow;
constexpr unsigned int SmemMask = (SmemSize * SmemStages - 1);

class SharedStorage {
  public:
    Array<int, SmemSize*SmemStages> array;
};

template <typename ThreadBlockShape_, int PitBlockShape_>
class IndexIterator {

  public:
    using Layout = layout::PitchLinear;
    using BytePointer = char *;
    using TensorCoord = Layout::TensorCoord;
    static int const PitBlockShape = PitBlockShape_;

    static int const kBlocksPerTiles = ThreadBlockShape_::kContiguous / PitBlockShape_;
    static int const kTilesPerLoad = SmemSize / kBlocksPerTiles;

  private:
    const int *gmem_ptr_;
    int *smem_ptr_;
    int kBlocksPerRow; // number of PIT blocks in a row
    int tile_offset_; // along the contiguous axis
    int block_offset_; // along the contiguous axis
    int smem_stage_;

  public:
    CUTLASS_DEVICE
    IndexIterator(
      SharedStorage& shared_storage_base,
      const int* gmem_ptr,
      TensorCoord extent, // of the sparse tensor
      TensorCoord const &threadblock_offset, // of the sparse tensor
      int thread_id)
      : smem_ptr_(reinterpret_cast<int*>(&shared_storage_base.array)),
        gmem_ptr_(const_cast<int*>(gmem_ptr)) {

      kBlocksPerRow = (extent.contiguous() + PitBlockShape - 1) / PitBlockShape;
      gmem_ptr_ += threadblock_offset.strided() * kBlocksPerRow;
      tile_offset_ = 0;
      block_offset_ = 0;
      smem_stage_ = 0;
      load_indices(0);
      __syncthreads();
      // if (threadIdx.x == 0) {
      //   printf(
      //     "gptr = [%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d]\n",
      //     gmem_ptr_[0],
      //     gmem_ptr_[1],
      //     gmem_ptr_[2],
      //     gmem_ptr_[3],
      //     gmem_ptr_[4],
      //     gmem_ptr_[5],
      //     gmem_ptr_[6],
      //     gmem_ptr_[7],
      //     gmem_ptr_[8],
      //     gmem_ptr_[9],
      //     gmem_ptr_[10],
      //     gmem_ptr_[11],
      //     gmem_ptr_[12],
      //     gmem_ptr_[13],
      //     gmem_ptr_[14],
      //     gmem_ptr_[15]
      //   );
      // }
    }

    CUTLASS_DEVICE
    void load_indices(int target_tile_offset) {
      if (tile_offset_ <= target_tile_offset && block_offset_ < kBlocksPerRow) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = threadIdx.x; i < SmemSize; i += blockDim.x) {
          if (block_offset_ + i < kBlocksPerRow) {
            smem_ptr_[smem_stage_ * SmemSize + i] = gmem_ptr_[block_offset_ + i];
          }
        }
        smem_stage_ ^= 1;
        tile_offset_ += kTilesPerLoad;
        block_offset_ += SmemSize;
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
    int get_advance_offset(int advance) {
      int x = advance / PitBlockShape;
      // int block_idx = smem_ptr_[x & SmemMask];
      int block_idx = gmem_ptr_[x];
      // std::printf(
      //   "[bid = %d, tid = %d] x = %d -> %d -> %d\n",
      //   int(blockIdx.x),
      //   int(threadIdx.x),
      //   int(advance),
      //   int(x),
      //   int(block_idx)
      // );
      return (block_idx - x) * PitBlockShape;
    }
};

}
}
}
}
