#pragma once

#include "cutlass/array.h"
#include "cutlass/coord.h"
#include "cutlass/cutlass.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/layout/pitch_linear.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/predicate_vector.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/tensor_view.h"

////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace transform {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

/// PitPredicatedTileAccessIterator
///
template <typename PitIndexIterator, typename Shape, typename Element, typename Layout, int AdvanceRank,
          typename ThreadMap, typename AccessType>
class PitPredicatedTileAccessIterator;

////////////////////////////////////////////////////////////////////////////////

/// Specialization of PitPredicatedTileAccessIterator for pitch-linear data.
///
template <typename PitIndexIterator, typename Shape_, typename Element_, int AdvanceRank,
          typename ThreadMap_, typename AccessType_>
class PitPredicatedTileAccessIterator<PitIndexIterator, Shape_, Element_, layout::PitchLinear,
                                   AdvanceRank, ThreadMap_, AccessType_> {
 public:
  static_assert(
      AdvanceRank == 0 || AdvanceRank == 1,
      "Specialization for pitch-linear iterator may along advance along the "
      "contiguous(rank=0) or strided(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::PitchLinear;
  static int const kAdvanceRank = AdvanceRank;
  using ThreadMap = ThreadMap_;
  using AccessType = AccessType_;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorView = TensorView<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using Pointer = Element *;
  using NonConstPointer = typename platform::remove_const<Element>::type *;

  static int const kAccessesPerVector = ThreadMap::kElementsPerAccess / AccessType::kElements;

  static_assert(!(ThreadMap::kElementsPerAccess % AccessType::kElements),
    "Vectors implied by the thread map must be divisible by the access type.");

  static int const kPredicatesPerByte = 4;
  static int const kPredicatesPerWord = 4 * kPredicatesPerByte;

  static int const kPredicateCount = ThreadMap::Iterations::kCount * kAccessesPerVector;

  /// Number of 32b words containing predicates
  static int const kPredicateByteCount =
    (kPredicateCount + kPredicatesPerByte - 1) / kPredicatesPerByte;
  static int const kPredicateWordCount = (kPredicateByteCount + 3) / 4;

  static unsigned const kPredicateMask = (1u << kPredicatesPerByte) - 1u;

  static_assert(kPredicateWordCount <= 4, "Too many predicates.");

  /// Predicate vector stores mask to guard accesses
  using Mask = Array<uint32_t, kPredicateWordCount>;

  /// Parameters object is precomputed state and is host-constructible
  class Params {
   public:
    friend PitPredicatedTileAccessIterator;

   private:
    /// stride of pitch-linear layout (units of Element)
    LongIndex stride_;
    /// amount (in byte) to increment pointer to move to next access along
    /// strided dimension
    LongIndex inc_strided_;
    /// amount (in byte) to increment pointer from last access to first access
    /// of next tile
    LongIndex inc_next_;
    /// amount (in byte) to increment pointer from first access of current tile
    /// to first access of next tile
    LongIndex inc_advance_;

   public:

    // Default ctor
    CUTLASS_HOST_DEVICE
    Params(): stride_(0), inc_strided_(0), inc_next_(0), inc_advance_(0) { }

    /// Construct the Params object given a pitch-linear tensor's layout
    CUTLASS_HOST_DEVICE
    Params(Layout const &layout) : stride_(layout.stride(0)) {
      inc_strided_ = (LongIndex(stride_) * ThreadMap::Delta::kStrided);

      if (kAdvanceRank) {
        // advance along strided dimension
        inc_advance_ = Shape::kStrided * LongIndex(stride_);
      } else {
        // advance along contiguous dimension
        inc_advance_ = Shape::kContiguous;
      }

      inc_next_ = inc_advance_ - LongIndex(ThreadMap::Iterations::kStrided - 1) * inc_strided_;
    };
  };

 private:
  /// Internal pointer type permits fast address arithmetic
  using BytePointer = char *;

 private:
  //
  // Data members
  //

  /// Parameters object with precomputed internal state
  Params const &params_;

  /// Internal pointer to first access of tile
  BytePointer pointer_;
  int pointer_offset_;

  /// PIT Index Iterator
  PitIndexIterator pit_index_iterator_;

  /// Guard predicates
  uint32_t predicates_[kPredicateWordCount];

  /// Size of tensor
  TensorCoord extent_;

  /// Initial offset for each thread
  TensorCoord thread_offset_;

  /// Offset to the first steady-state tile
  TensorCoord residue_offset_;

  /// Used for out-of-order visitation
  bool is_residue_tile_;

  /// Iteration along vectors implied by the thread map
  int iteration_vector_;

  /// Iteration in the contiguous dimension
  int iteration_contiguous_;

  /// Iteration in the strided dimension
  int iteration_strided_;

  /// Number of iterated tiles
  int tile_offset_;

 public:
  /// Computes predicates based on internally tracked per-thread offset.
  CUTLASS_DEVICE
  void compute_predicates_(
      /// Extent of the matrix window
      TensorCoord extent,
      /// optionally, simplify predicate calculation during 'steady state' phase
      bool is_steady_state = false) {

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kPredicateWordCount; ++i) {
      predicates_[i] = 0u;
    }

    CUTLASS_PRAGMA_UNROLL
    for (int access_idx = 0; access_idx < ThreadMap::Iterations::kCount * kAccessesPerVector; ++access_idx) {

      int s = access_idx / (ThreadMap::Iterations::kContiguous * kAccessesPerVector);
      
      int access_residual = access_idx % (ThreadMap::Iterations::kContiguous * kAccessesPerVector);

      int c = access_residual / kAccessesPerVector;
      int v = access_residual % kAccessesPerVector;

      TensorCoord iteration_coord(c * ThreadMap::Delta::kContiguous + v * AccessType::kElements,
                                s * ThreadMap::Delta::kStrided);

      TensorCoord coord = thread_offset_ + iteration_coord;

      bool guard;

      if (is_steady_state) {
        if (kAdvanceRank == 0) {
          guard = (coord.strided() < extent.strided());
        } else {
          guard = (coord.contiguous() < extent.contiguous());
        }
      } else {
        guard = (coord.strided() < extent.strided() &&
                 coord.contiguous() < extent.contiguous());
      }

      int pred_idx = v + kAccessesPerVector * (c + ThreadMap::Iterations::kContiguous * s);

      int word_idx = pred_idx / kPredicatesPerWord;
      int residual = pred_idx % kPredicatesPerWord;
      int byte_idx = residual / kPredicatesPerByte;
      int bit_idx = residual % kPredicatesPerByte;
      
      predicates_[word_idx] |= (unsigned(guard) << (byte_idx * 8 + bit_idx));

    }

  }

 public:
  /// Constructs a TileIterator from its precomputed state, threadblock offset,
  /// and thread ID
  CUTLASS_HOST_DEVICE
  PitPredicatedTileAccessIterator(
      /// Precomputed parameters object
      Params const &params,
      /// Pointer to start of tensor
      Pointer pointer,
      /// Extent of tensor
      TensorCoord extent,
      /// ID of each participating thread
      int thread_id,
      /// Initial offset of threadblock
      TensorCoord const &threadblock_offset,
      PitIndexIterator index_iterator)
      : params_(params),
        pointer_(reinterpret_cast<BytePointer>(
            const_cast<NonConstPointer>(pointer))),
        pointer_offset_(0),
        tile_offset_(0),
        extent_(extent),
        is_residue_tile_(true),
        pit_index_iterator_(index_iterator) {
          
    TensorCoord residue_extent;
    if (kAdvanceRank) {

      typename TensorCoord::Index residue_size = (extent_[kAdvanceRank] - threadblock_offset.strided()) % Shape::kStrided;
      if (!residue_size) {
        residue_size = Shape::kStrided;
      }

      residue_offset_ = make_Coord(0, residue_size);
      residue_extent = make_Coord(
        extent_.contiguous(), 
        min(threadblock_offset.strided() + residue_size, extent_.strided())
      );
    } else {

      typename TensorCoord::Index residue_size = (extent_[kAdvanceRank] - threadblock_offset.contiguous()) % Shape::kContiguous;
      if (!residue_size) {
        residue_size = Shape::kContiguous;
      }

      residue_offset_ = make_Coord(residue_size, 0);
      
      residue_extent = make_Coord(
        min(extent_.contiguous(), threadblock_offset.contiguous() + residue_size),
        extent_.strided()
      );
    }

    // Per-thread offset in logical coordinates of tensor
    thread_offset_ = threadblock_offset + ThreadMap::initial_offset(thread_id);

    // update internal pointers
    Layout layout(params_.stride_);
    add_pointer_offset(layout(thread_offset_));

    compute_predicates_(residue_extent, false);

    set_iteration_index(0);

    // if (threadIdx.x == 0) {
    //   std::printf(
    //     "[bid = (%d, %d, %d)] extent = (%d, %d), tb_offset = (%d, %d)\n",
    //     int(blockIdx.x), int(blockIdx.y), int(blockIdx.z),
    //     int(extent_.strided()),
    //     int(extent_.contiguous()),
    //     int(threadblock_offset.strided()),
    //     int(threadblock_offset.contiguous())
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

  /// Construct a PitPredicatedTileAccessIterator with zero threadblock offset
  CUTLASS_HOST_DEVICE
  PitPredicatedTileAccessIterator(
      /// Precomputed parameters object
      Params const &params,
      /// Pointer to start of tensor
      Pointer pointer,
      /// Extent of tensor
      TensorCoord extent,
      ///< ID of each participating thread
      int thread_id,
      PitIndexIterator index_iterator)
      : PitPredicatedTileAccessIterator(params, pointer, extent, thread_id,
                                     make_Coord(0, 0), index_iterator) {}

  /// Overrides the internal iteration index
  CUTLASS_HOST_DEVICE
  void set_iteration_index(int index) {

    iteration_vector_ = index % kAccessesPerVector;
    int residual_access = index / kAccessesPerVector;

    iteration_contiguous_ = residual_access % ThreadMap::Iterations::kContiguous;
    iteration_strided_ = residual_access / ThreadMap::Iterations::kContiguous;

  }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    pointer_offset_ += pointer_offset;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  CUTLASS_DEVICE
  void add_tile_offset(
      TensorCoord const &tile_offset) {
    if (is_residue_tile_) {

      thread_offset_ += residue_offset_;

      Layout layout(params_.stride_);
      add_pointer_offset(layout(residue_offset_));

      compute_predicates_(extent_, true);

      if (kAdvanceRank) { // K: strided
        pointer_offset_ += params_.inc_advance_ * LongIndex(tile_offset.strided() - 1);
        pointer_offset_ += Shape::kContiguous * tile_offset.contiguous(); // * 8 / sizeof_bits<Element>::value
      } else {
        pointer_offset_ += params_.inc_advance_ * LongIndex(tile_offset.contiguous() - 1);
        pointer_offset_ += Shape::kStrided * tile_offset.strided(); // * 8 / sizeof_bits<Element>::value
      }
    } else {
      if (kAdvanceRank) {
        pointer_offset_ += params_.inc_advance_ * LongIndex(tile_offset.strided());
        pointer_offset_ += Shape::kContiguous * tile_offset.contiguous(); // * 8 / sizeof_bits<Element>::value
      } else {
        pointer_offset_ += params_.inc_advance_ * LongIndex(tile_offset.contiguous());
        pointer_offset_ += Shape::kStrided * tile_offset.strided(); // * 8 / sizeof_bits<Element>::value
      }
    }
    is_residue_tile_ = false;
    if (kAdvanceRank) {
      tile_offset_ += tile_offset.strided();
    } else {
      tile_offset_ += tile_offset.contiguous();
    }
    pit_index_iterator_.load_indices(tile_offset_);
  }

  /// Returns a pointer
  // CUTLASS_HOST_DEVICE
  // AccessType *get() const {
  //   return reinterpret_cast<AccessType *>(pointer_ + (
  //     pointer_offset_ + iteration_contiguous_ * ThreadMap::Delta::kContiguous + iteration_vector_
  //   ) * sizeof_bits<Element>::value / 8);
  // }

  /// Returns a pointer of current vector
  CUTLASS_DEVICE
  AccessType *get() {
    int real_offset = pit_index_iterator_.get_real_offset(
      pointer_offset_ + iteration_contiguous_ * ThreadMap::Delta::kContiguous + iteration_vector_
    );
    if (real_offset < 0) {
      return reinterpret_cast<AccessType *>(pit_index_iterator_.get_zero_pointer());
    } else {
      return reinterpret_cast<AccessType *>(pointer_ + real_offset * sizeof_bits<Element>::value / 8);
    }
  }

  CUTLASS_HOST_DEVICE
  int get_stride() const {
    if(kAdvanceRank)
      return params_.stride_;
    else
      return 1;
  }

  /// Increment and return an instance to self.
  CUTLASS_HOST_DEVICE
  PitPredicatedTileAccessIterator &operator++() {

    ++iteration_vector_;
    if (iteration_vector_ < kAccessesPerVector) {
      return *this;
    }

    iteration_vector_ = 0;
    ++iteration_contiguous_;

    if (iteration_contiguous_ < ThreadMap::Iterations::kContiguous) {
      return *this;
    }

    // Enter here only if (iteration_contiguous_ ==
    // ThreadMap::Iteration::kContiguous)
    iteration_contiguous_ = 0;
    ++iteration_strided_;

    if (iteration_strided_ < ThreadMap::Iterations::kStrided) {
      pointer_offset_ += params_.inc_strided_;
      return *this;
    }

    // Enter here only if (iteration_stride_ == ThreadMap::Iteration::kStrided)
    // which means we enter the next tile.
    iteration_strided_ = 0;

    // advance to next tile
    pointer_offset_ += params_.inc_next_;

    // now return to start tile - if the iterator is subsequently advanced, this
    // subtraction as well as the subsequent integer addition are both elided by
    // the compiler.
    pointer_offset_ -= params_.inc_advance_;

    return *this;
  }

  /// Increment and return an instance to self.
  CUTLASS_HOST_DEVICE
  PitPredicatedTileAccessIterator operator++(int) {
    PitPredicatedTileAccessIterator self(*this);
    operator++();
    return self;
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void clear_mask(bool enable = true) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kPredicateWordCount; ++i) {
      predicates_[i] = enable ? 0u : predicates_[i];
    }

  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void enable_mask() {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kPredicateWordCount; ++i) {
      predicates_[i] = 0xffffffff;
    }

  }

  /// Sets the predicate mask, overriding value stored in predicate iterator
  CUTLASS_HOST_DEVICE
  void set_mask(Mask const &mask) { 
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kPredicateWordCount; ++i) {
      predicates_[i] = mask[i];
    }

  }

  /// Gets the mask
  CUTLASS_HOST_DEVICE
  void get_mask(Mask &mask) {
     CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kPredicateWordCount; ++i) {
      mask[i] = predicates_[i];
    }
  }

  /// Returns whether access is valid or not
  CUTLASS_HOST_DEVICE
  bool valid() {

    int pred_idx = 
      iteration_vector_ + kAccessesPerVector * (iteration_contiguous_ + iteration_strided_ * ThreadMap::Iterations::kContiguous);

    int word_idx = pred_idx / kPredicatesPerWord;
    int residual = pred_idx % kPredicatesPerWord;
    int byte_idx = residual / kPredicatesPerByte;
    int bit_idx = residual % kPredicatesPerByte;
    
    bool pred = (predicates_[word_idx] & (1u << (byte_idx * 8 + bit_idx))) != 0;
    return pred;
    
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Specialization of PitPredicatedTileAccessIterator for pitch-linear data.
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept |
///            MaskedTileIteratorConcept
///
template <typename PitIndexIterator, typename Shape_, typename Element_, int AdvanceRank,
          typename ThreadMap_, typename AccessType_>
class PitPredicatedTileAccessIterator<PitIndexIterator, Shape_, Element_, layout::ColumnMajor,
                                   AdvanceRank, ThreadMap_, AccessType_> {
 public:
  static_assert(
      AdvanceRank == 0 || AdvanceRank == 1,
      "Specialization for pitch-linear iterator may along advance along the "
      "contiguous(rank=0) or strided(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::ColumnMajor;
  static int const kAdvanceRank = AdvanceRank;
  using ThreadMap = ThreadMap_;
  using AccessType = AccessType_;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorView = TensorView<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using Pointer = Element *;
  using NonConstPointer = typename platform::remove_const<Element>::type *;

  using UnderlyingIterator = PitPredicatedTileAccessIterator<
      PitIndexIterator,
      layout::PitchLinearShape<Shape::kRow, Shape::kColumn>, Element,
      layout::PitchLinear, (kAdvanceRank == 0 ? 0 : 1), ThreadMap, AccessType>;

  /// Predicate vector stores mask to guard accesses
  using Mask = typename UnderlyingIterator::Mask;

  static int const kAccessesPerVector = UnderlyingIterator::kAccessesPerVector;

  /// Parameters object is precomputed state and is host-constructible
  class Params {
   private:
    friend PitPredicatedTileAccessIterator;

    /// Parameters object
    typename UnderlyingIterator::Params params_;

   public:

    /// Default ctor
    CUTLASS_HOST_DEVICE
    Params() { }

    /// Construct the Params object given a pitch-linear tensor's layout
    CUTLASS_HOST_DEVICE
    Params(Layout const &layout)
        : params_(layout::PitchLinear(layout.stride(0))){};
  };

 private:
  //
  // Data members
  //

  /// Underlying pitch-linear tile iterator
  UnderlyingIterator iterator_;

 public:
  /// Constructs a TileIterator from its precomputed state, threadblock offset,
  /// and thread ID
  CUTLASS_HOST_DEVICE
  PitPredicatedTileAccessIterator(
      ///< Precomputed parameters object
      Params const &params,
      ///< Pointer to start of tensor
      Pointer pointer,
      ///< Extent of tensor
      TensorCoord extent,
      ///< ID of each participating thread
      int thread_id,
      ///< Initial offset of threadblock
      TensorCoord const &threadblock_offset,
      PitIndexIterator index_iterator)
      : iterator_(params.params_, pointer,
                  layout::PitchLinearCoord(extent.row(), extent.column()),
                  thread_id,
                  layout::PitchLinearCoord(threadblock_offset.row(),
                                           threadblock_offset.column()),
                  index_iterator) {}

  /// Construct a PitPredicatedTileAccessIterator with zero threadblock offset
  CUTLASS_HOST_DEVICE
  PitPredicatedTileAccessIterator(
      Params const &params,  ///< Precomputed parameters object
      Pointer pointer,       ///< Pointer to start of tensor
      TensorCoord extent,    ///< Extent of tensor
      int thread_id,         ///< ID of each participating thread
      PitIndexIterator index_iterator)
      : PitPredicatedTileAccessIterator(params, pointer, extent, thread_id,
                                     make_Coord(0, 0), index_iterator) {}

  /// Overrides the internal iteration index
  CUTLASS_HOST_DEVICE
  void set_iteration_index(int index) { iterator_.set_iteration_index(index); }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  CUTLASS_HOST_DEVICE
  void add_tile_offset(TensorCoord const &tile_offset) {
    iterator_.add_tile_offset({tile_offset.row(), tile_offset.column()});
  }

  /// Returns a pointer
  CUTLASS_DEVICE
  AccessType *get() {
    return reinterpret_cast<AccessType *>(iterator_.get());
  }
  
  CUTLASS_HOST_DEVICE
  int get_stride() const {
    return iterator_.get_stride();
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  PitPredicatedTileAccessIterator &operator++() {
    ++iterator_;
    return *this;
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  PitPredicatedTileAccessIterator operator++(int) {
    PitPredicatedTileAccessIterator self(*this);
    operator++();
    return self;
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void clear_mask(bool enable = true) { iterator_.clear_mask(enable); }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void enable_mask() { iterator_.enable_mask(); }

  /// Sets the predicate mask, overriding value stored in predicate iterator
  CUTLASS_HOST_DEVICE
  void set_mask(Mask const &mask) { iterator_.set_mask(mask); }

  /// Gets the mask
  CUTLASS_HOST_DEVICE
  void get_mask(Mask &mask) { iterator_.get_mask(mask); }

  /// Returns whether access is valid or not
  CUTLASS_HOST_DEVICE
  bool valid() {
    return iterator_.valid();
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Specialization of PitPredicatedTileAccessIterator for pitch-linear data.
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept |
///            MaskedTileIteratorConcept
///
template <typename PitIndexIterator, typename Shape_, typename Element_, int AdvanceRank,
          typename ThreadMap_, typename AccessType_>
class PitPredicatedTileAccessIterator<PitIndexIterator, Shape_, Element_, layout::RowMajor,
                                   AdvanceRank, ThreadMap_, AccessType_> {
 public:
  static_assert(
      AdvanceRank == 0 || AdvanceRank == 1,
      "Specialization for pitch-linear iterator may along advance along the "
      "contiguous(rank=0) or strided(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::RowMajor;
  static int const kAdvanceRank = AdvanceRank;
  using ThreadMap = ThreadMap_;
  using AccessType = AccessType_;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorView = TensorView<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using Pointer = Element *;
  using NonConstPointer = typename platform::remove_const<Element>::type *;

  using UnderlyingIterator = PitPredicatedTileAccessIterator<
      PitIndexIterator,
      layout::PitchLinearShape<Shape::kColumn, Shape::kRow>, Element,
      layout::PitchLinear, (kAdvanceRank == 0 ? 1 : 0), ThreadMap, AccessType>;

  static int const kAccessesPerVector = UnderlyingIterator::kAccessesPerVector;

  /// Predicate vector stores mask to guard accesses
  using Mask = typename UnderlyingIterator::Mask;

  /// Parameters object is precomputed state and is host-constructible
  class Params {
   private:
    friend PitPredicatedTileAccessIterator;

    /// Parameters object
    typename UnderlyingIterator::Params params_;

   public:

    /// Default ctor
    CUTLASS_HOST_DEVICE
    Params() { }

    /// Construct the Params object given a pitch-linear tensor's layout
    CUTLASS_HOST_DEVICE
    Params(Layout const &layout)
        : params_(layout::PitchLinear(layout.stride(0))){};
  };

 private:
  //
  // Data members
  //

  /// Underlying pitch-linear tile iterator
  UnderlyingIterator iterator_;

 public:
  /// Constructs a TileIterator from its precomputed state, threadblock offset,
  /// and thread ID
  CUTLASS_HOST_DEVICE
  PitPredicatedTileAccessIterator(
      ///< Precomputed parameters object
      Params const &params,
      ///< Pointer to start of tensor
      Pointer pointer,
      ///< Extent of tensor
      TensorCoord extent,
      ///< ID of each participating thread
      int thread_id,
      ///< Initial offset of threadblock
      TensorCoord const &threadblock_offset,
      PitIndexIterator index_iterator)
      : iterator_(params.params_, pointer,
                  layout::PitchLinearCoord(extent.column(), extent.row()),
                  thread_id,
                  layout::PitchLinearCoord(threadblock_offset.column(),
                                           threadblock_offset.row()),
                  index_iterator) {}

  /// Construct a PitPredicatedTileAccessIterator with zero threadblock offset
  CUTLASS_HOST_DEVICE
  PitPredicatedTileAccessIterator(
      Params const &params,  ///< Precomputed parameters object
      Pointer pointer,       ///< Pointer to start of tensor
      TensorCoord extent,    ///< Extent of tensor
      int thread_id,         ///< ID of each participating thread
      PitIndexIterator index_iterator)
      : PitPredicatedTileAccessIterator(params, pointer, extent, thread_id,
                                     make_Coord(0, 0), index_iterator) {}

  /// Overrides the internal iteration index
  CUTLASS_HOST_DEVICE
  void set_iteration_index(int index) { iterator_.set_iteration_index(index); }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  CUTLASS_HOST_DEVICE
  void add_tile_offset(TensorCoord const &tile_offset) {
    iterator_.add_tile_offset({tile_offset.column(), tile_offset.row()});
  }

  /// Returns a pointer
  CUTLASS_DEVICE
  AccessType *get() {
    return reinterpret_cast<AccessType *>(iterator_.get());
  }
  
  CUTLASS_HOST_DEVICE
  int get_stride() const {
    return iterator_.get_stride();
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  PitPredicatedTileAccessIterator &operator++() {
    ++iterator_;
    return *this;
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  PitPredicatedTileAccessIterator operator++(int) {
    PitPredicatedTileAccessIterator self(*this);
    operator++();
    return self;
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void clear_mask(bool enable = true) { iterator_.clear_mask(enable); }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void enable_mask() { iterator_.enable_mask(); }

  /// Sets the predicate mask, overriding value stored in predicate iterator
  CUTLASS_HOST_DEVICE
  void set_mask(Mask const &mask) { iterator_.set_mask(mask); }

  /// Gets the mask
  CUTLASS_HOST_DEVICE
  void get_mask(Mask &mask) { iterator_.get_mask(mask); }
  
  /// Returns whether access is valid or not
  CUTLASS_HOST_DEVICE
  bool valid() {
    return iterator_.valid();
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Specialization of PitPredicatedTileAccessIterator for column-major interleaved data.
/// It is mapped to the congruous layout.
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept |
///            MaskedTileIteratorConcept
///

template <typename PitIndexIterator, typename Shape_, typename Element_, int AdvanceRank,
          typename ThreadMap_, typename AccessType_, int InterleavedK>
class PitPredicatedTileAccessIterator<PitIndexIterator, Shape_, Element_,
                                   layout::ColumnMajorInterleaved<InterleavedK>,
                                   AdvanceRank, ThreadMap_, AccessType_> {
 public:
  static_assert(
      AdvanceRank == 0 || AdvanceRank == 1,
      "Specialization for pitch-linear iterator may along advance along the "
      "contiguous(rank=0) or strided(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  static int const kInterleavedK = InterleavedK;
  using Layout = layout::ColumnMajorInterleaved<kInterleavedK>;
  static int const kAdvanceRank = AdvanceRank;
  using ThreadMap = ThreadMap_;
  using AccessType = AccessType_;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorView = TensorView<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using Pointer = Element *;
  using NonConstPointer = typename platform::remove_const<Element>::type *;

  using UnderlyingIterator = PitPredicatedTileAccessIterator<
      PitIndexIterator,
      layout::PitchLinearShape<Shape::kRow * kInterleavedK,
                               Shape::kColumn / kInterleavedK>,
      Element, layout::PitchLinear, (kAdvanceRank == 0 ? 0 : 1), ThreadMap,
      AccessType>;

  static int const kAccessesPerVector = UnderlyingIterator::kAccessesPerVector;

  /// Predicate vector stores mask to guard accesses
  using Mask = typename UnderlyingIterator::Mask;

  /// Parameters object is precomputed state and is host-constructible
  class Params {
   private:
    friend PitPredicatedTileAccessIterator;

    /// Parameters object
    typename UnderlyingIterator::Params params_;

   public:
    CUTLASS_HOST_DEVICE
    Params() {}

    /// Construct the Params object given a pitch-linear tensor's layout
    CUTLASS_HOST_DEVICE
    Params(Layout const &layout)
        : params_(layout::PitchLinear(layout.stride(0))) {}
  };

 private:
  //
  // Data members
  //

  /// Underlying pitch-linear tile iterator
  UnderlyingIterator iterator_;

 public:
  /// Constructs a TileIterator from its precomputed state, threadblock offset,
  /// and thread ID
  CUTLASS_HOST_DEVICE
  PitPredicatedTileAccessIterator(
      /// Precomputed parameters object
      Params const &params,
      /// Pointer to start of tensor
      Pointer pointer,
      /// Extent of tensor
      TensorCoord extent,
      /// ID of each participating thread
      int thread_id,
      /// Initial offset of threadblock
      TensorCoord const &threadblock_offset,
      PitIndexIterator index_iterator)
      : iterator_(params.params_, pointer,
                  layout::PitchLinearCoord(extent.row() * kInterleavedK,
                                           extent.column() / kInterleavedK),
                  thread_id,
                  layout::PitchLinearCoord(
                      threadblock_offset.row() * kInterleavedK,
                      threadblock_offset.column() / kInterleavedK),
                  index_iterator) {}

  /// Construct a PitPredicatedTileAccessIterator with zero threadblock offset
  CUTLASS_HOST_DEVICE
  PitPredicatedTileAccessIterator(
      Params const &params,  ///< Precomputed parameters object
      Pointer pointer,       ///< Pointer to start of tensor
      TensorCoord extent,    ///< Extent of tensor
      int thread_id,         ///< ID of each participating thread
      PitIndexIterator index_iterator)
      : PitPredicatedTileAccessIterator(params, pointer, extent, thread_id,
                                     make_Coord(0, 0), index_iterator) {}

  /// Overrides the internal iteration index
  CUTLASS_HOST_DEVICE
  void set_iteration_index(int index) { iterator_.set_iteration_index(index); }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  CUTLASS_HOST_DEVICE
  void add_tile_offset(TensorCoord const &tile_offset) {
    iterator_.add_tile_offset({tile_offset.row(), tile_offset.column()});
  }

  /// Returns a pointer
  CUTLASS_DEVICE
  AccessType *get() {
    return reinterpret_cast<AccessType *>(iterator_.get());
  }
  
  CUTLASS_HOST_DEVICE
  int get_stride() const {
    return iterator_.get_stride();
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  PitPredicatedTileAccessIterator &operator++() {
    ++iterator_;
    return *this;
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  PitPredicatedTileAccessIterator operator++(int) {
    PitPredicatedTileAccessIterator self(*this);
    operator++();
    return self;
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void clear_mask(bool enable = true) { iterator_.clear_mask(enable); }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void enable_mask() { iterator_.enable_mask(); }

  /// Sets the predicate mask, overriding value stored in predicate iterator
  CUTLASS_HOST_DEVICE
  void set_mask(Mask const &mask) { iterator_.set_mask(mask); }

  /// Gets the mask
  CUTLASS_HOST_DEVICE
  void get_mask(Mask &mask) { iterator_.get_mask(mask); }

  /// Returns whether access is valid or not
  CUTLASS_HOST_DEVICE
  bool valid() { return iterator_.valid(); }
};

////////////////////////////////////////////////////////////////////////////////

/// Specialization of PitPredicatedTileAccessIterator for row-major interleaved data.
/// It is mapped to the congruous layout.
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept |
///            MaskedTileIteratorConcept
///
template <typename PitIndexIterator, typename Shape_, typename Element_, int AdvanceRank,
          typename ThreadMap_, typename AccessType_, int InterleavedK>
class PitPredicatedTileAccessIterator<PitIndexIterator, Shape_, Element_,
                                   layout::RowMajorInterleaved<InterleavedK>,
                                   AdvanceRank, ThreadMap_, AccessType_> {
 public:
  static_assert(
      AdvanceRank == 0 || AdvanceRank == 1,
      "Specialization for pitch-linear iterator may along advance along the "
      "contiguous(rank=0) or strided(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  static int const kInterleavedK = InterleavedK;
  using Layout = layout::RowMajorInterleaved<kInterleavedK>;
  static int const kAdvanceRank = AdvanceRank;
  using ThreadMap = ThreadMap_;
  using AccessType = AccessType_;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorView = TensorView<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using Pointer = Element *;
  using NonConstPointer = typename platform::remove_const<Element>::type *;

  using UnderlyingIterator = PitPredicatedTileAccessIterator<
      PitIndexIterator,
      layout::PitchLinearShape<Shape::kColumn * kInterleavedK,
                               Shape::kRow / kInterleavedK>,
      Element, layout::PitchLinear, (kAdvanceRank == 0 ? 1 : 0), ThreadMap,
      AccessType>;


  static int const kAccessesPerVector = UnderlyingIterator::kAccessesPerVector;

  /// Predicate vector stores mask to guard accesses
  using Mask = typename UnderlyingIterator::Mask;

  /// Parameters object is precomputed state and is host-constructible
  class Params {
   private:
    friend PitPredicatedTileAccessIterator;

    /// Parameters object
    typename UnderlyingIterator::Params params_;

   public:
    CUTLASS_HOST_DEVICE
    Params() {}

    /// Construct the Params object given a pitch-linear tensor's layout
    CUTLASS_HOST_DEVICE
    Params(Layout const &layout)
        : params_(layout::PitchLinear(layout.stride(0))) {}
  };

 private:
  //
  // Data members
  //

  /// Underlying pitch-linear tile iterator
  UnderlyingIterator iterator_;

 public:
  /// Constructs a TileIterator from its precomputed state, threadblock offset,
  /// and thread ID
  CUTLASS_HOST_DEVICE
  PitPredicatedTileAccessIterator(
      /// Precomputed parameters object
      Params const &params,
      /// Pointer to start of tensor
      Pointer pointer,
      /// Extent of tensor
      TensorCoord extent,
      /// ID of each participating thread
      int thread_id,
      /// Initial offset of threadblock
      TensorCoord const &threadblock_offset,
      PitIndexIterator index_iterator)
      : iterator_(params.params_, pointer,
                  layout::PitchLinearCoord(extent.column() * kInterleavedK,
                                           extent.row() / kInterleavedK),
                  thread_id,
                  layout::PitchLinearCoord(
                      threadblock_offset.column() * kInterleavedK,
                      threadblock_offset.row() / kInterleavedK),
                  index_iterator) {}

  /// Construct a PitPredicatedTileAccessIterator with zero threadblock offset
  CUTLASS_HOST_DEVICE
  PitPredicatedTileAccessIterator(
      Params const &params,  ///< Precomputed parameters object
      Pointer pointer,       ///< Pointer to start of tensor
      TensorCoord extent,    ///< Extent of tensor
      int thread_id,         ///< ID of each participating thread
      PitIndexIterator index_iterator)
      : PitPredicatedTileAccessIterator(params, pointer, extent, thread_id,
                                     make_Coord(0, 0), index_iterator) {}

  /// Overrides the internal iteration index
  CUTLASS_HOST_DEVICE
  void set_iteration_index(int index) { iterator_.set_iteration_index(index); }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  CUTLASS_HOST_DEVICE
  void add_tile_offset(TensorCoord const &tile_offset) {
    iterator_.add_tile_offset({tile_offset.column(), tile_offset.row()});
  }

  /// Returns a pointer
  CUTLASS_DEVICE
  AccessType *get() {
    return reinterpret_cast<AccessType *>(iterator_.get());
  }
  
  CUTLASS_HOST_DEVICE
  int get_stride() const {
    return iterator_.get_stride();
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  PitPredicatedTileAccessIterator &operator++() {
    ++iterator_;
    return *this;
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  PitPredicatedTileAccessIterator operator++(int) {
    PitPredicatedTileAccessIterator self(*this);
    operator++();
    return self;
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void clear_mask(bool enable = true) { iterator_.clear_mask(enable); }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void enable_mask() { iterator_.enable_mask(); }

  /// Sets the predicate mask, overriding value stored in predicate iterator
  CUTLASS_HOST_DEVICE
  void set_mask(Mask const &mask) { iterator_.set_mask(mask); }

  /// Gets the mask
  CUTLASS_HOST_DEVICE
  void get_mask(Mask &mask) { iterator_.get_mask(mask); }

  /// Returns whether access is valid or not
  CUTLASS_HOST_DEVICE
  bool valid() { return iterator_.valid(); }
};

////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace transform
}  // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
