#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/arch/memory.h"
#include "cutlass/layout/layout.h"

namespace cutlass {
namespace transform {
namespace threadblock {

namespace pit {

constexpr unsigned int SmemPow = 9; // 9 for blockDim.x == 64
constexpr unsigned int SmemStages = 2;
constexpr unsigned int SmemSize = 1 << SmemPow;
constexpr unsigned int SmemMask = (SmemSize*SmemStages-1);

class SharedStorage {
  public:
    Array<int, SmemSize*SmemStages> array;
};

class IndexIterator {
  public:
    using AccessType = AlignedArray<int, 8>;

  private:
    const int *gmem_idx_;
    int *smem_idx_;
    AccessType *frag_ptr_;
    int offset_x_;
    int smem_stage_;
    const int matrix_size_x_;
    const int block_size_x_;
    const int block_size_y_;
    const int block_num_tb_x_;
    const int block_num_tb_y_;
    const int block_num_matrix_x_;
    const int block_num_smem_x_;
    const int tb_offset_y_;

  public:
    CUTLASS_DEVICE
    IndexIterator(
      SharedStorage& shared_storage_base,
      const int* idx,
      const int matrix_size_x,
      const int block_size_x,
      const int block_size_y,
      const int tb_size_x,
      const int tb_size_y,
      const int tb_idx_x,
      const int tb_idx_y
    ) : 
        smem_idx_(reinterpret_cast<int*>(&shared_storage_base.array)),
        gmem_idx_(const_cast<int*>(idx)),
        matrix_size_x_(matrix_size_x),
        block_size_x_(block_size_x),
        block_size_y_(block_size_y),
        block_num_tb_x_(tb_size_x / block_size_x),
        block_num_tb_y_(tb_size_y / block_size_y),
        block_num_matrix_x_((matrix_size_x_ + block_size_x_ - 1) / block_size_x_),
        block_num_smem_x_(SmemSize / block_num_tb_y_),
        tb_offset_y_(tb_idx_y * block_num_tb_y_) {
      int frag[8] = { 0 };
      frag_ptr_ = reinterpret_cast<AccessType *>(&frag);
      offset_x_ = 0;
      load_indices();
      __syncthreads();
    }

    CUTLASS_DEVICE
    int load_indices() {
      for (int i = threadIdx.x * 8; i < SmemSize; i += blockDim.x * 8) {
        int gidx_y = tb_offset_y_ + i / block_num_smem_x_;
        int gidx_x = offset_x_ + i % block_num_smem_x_;
        if (gidx_x > matrix_size_x_) break;
        AccessType const *access_ptr = reinterpret_cast<AccessType const *>(
          &gmem_idx_[gidx_y * block_num_matrix_x_ + gidx_x]);
        arch::global_load<AccessType, sizeof(AccessType), arch::CacheOperation::LastUse>(
          frag_ptr_[0], access_ptr, true);
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < 8; j++) {
          smem_idx_[smem_stage_ * SmemSize + i + j] = frag_ptr_->at(j);
        }
      }
      smem_stage_ ^= 1;
      offset_x_ += block_num_smem_x_;
    }

    CUTLASS_DEVICE
    int get_real_offset(const int& offset) {
      int row_idx = offset / matrix_size_x_;
      int col_idx = offset % matrix_size_x_;
      int sidx_y = row_idx / block_size_y_;
      int sidx_x = col_idx / block_size_x_;
      if (sidx_x >= offset_x_) {
        load_indices();
      }
      sidx_y += sidx_x / block_num_smem_x_ * block_num_tb_y_;
      sidx_x %= block_num_smem_x_;
      int sidx = sidx_y * block_num_smem_x_ + sidx_x;
      int real_col_idx = smem_idx_[sidx & SmemMask] * block_size_x_ + col_idx % block_size_x_;
      return row_idx * matrix_size_x_ + real_col_idx;
    }
};

}
}
}
}
