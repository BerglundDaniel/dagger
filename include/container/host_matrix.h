#pragma once

#include <cstdlib>
#include <cassert>

#include "cuda/cuda_adapter.h"
#include "cuda/cuda_types_concept.h"
#include "util/multiplier.h"
#include "container_properties.h"

namespace dagger::container {

/**
 * Matrix with memory stored on the host.
 * Pinned memory prevents paging. It is faster to transfer and required for async tranfers
 * Allocated amount of memory is a multiplier of lengthMultiplier.
 * The class is the owner of the memory unless it is a subcontainer.
 * Column major
 */
template <typename T, int loopUnroll>
requires cudaTypes<T> && (loopUnroll > 0)
class HostMatrix {
 public:
  explicit HostMatrix(int numberOfRows, int numberOfColumns, bool usePinnedMemory=true)
      : properties_{
          .numberOfRows = numberOfRows,
          .numberOfColumns = numberOfColumns,
          .leadingDimension = multiplier::calculateLeadingDimension(false, numberOfRows, loopUnroll)},
       use_pinned_memory_(usePinnedMemory), sub_container_(false), memory_ptr_(nullptr) {
    assert(numberOfRows > 0);
    assert(numberOfColumns > 0);
    if(use_pinned_memory_){
      allocateHostPinnedMemory<T>((void**) &(memory_ptr_),
					properties_.leadingDimension * properties_.numberOfColumns);
    } else {
      memory_ptr_ = (T*) malloc(sizeof(T) * properties_.leadingDimension * properties_.numberOfColumns);
    }
  }

  ~HostMatrix() {
    if(!sub_container_){
      if(use_pinned_memory_){
	freePinnedMemory(memory_ptr_);
      } else {
	free(memory_ptr_);
      }
    }
  }

  inline constexpr int loopUnrollMultiplier() {
    return loopUnroll;
  }

  inline constexpr int leadingDimensionMultiplier() const {
    return loopUnroll;
  }

  inline ContainerProperties properties() const {
    return properties_;
  }

  inline int numberOfRows() const {
    return properties_.numberOfRows;
  }

  inline int numberOfColumns() const {
    return properties_.numberOfColumns;
  }

  inline int leadingDimension() const {
    return properties_.leadingDimension;
  }

  inline int isSubContainer() const {
    return sub_container_;
  }

  inline int isPinnedMemory() const {
    return use_pinned_memory_;
  }

  inline T& operator()(int row, int column) {
    return *(memory_ptr_ + (properties_.leadingDimension * column) + row);
  }

  inline const T& operator()(int row, int column) const {
    return *(memory_ptr_ + (properties_.leadingDimension * column) + row);
  }

  inline T* memoryPointer() {
    return memory_ptr_;
  }

  inline const T* memoryPointer() const {
    return memory_ptr_;
  }

  HostMatrix(const HostMatrix&) = delete;
  HostMatrix(HostMatrix&&) = delete;
  HostMatrix& operator=(const HostMatrix&) = delete;
  HostMatrix& operator=(HostMatrix&&) = delete;

 private:
  //explicit HostMatrix(size_t numberOfRealRows, size_t numberOfRealColumns, size_t numberOfRows, size_t numberOfColumns, bool pinnedMemory=true, size_t loopUnroll=1);
  //explicit HostMatrix(size_t numberOfRealRows, size_t numberOfRealColumns, size_t numberOfRows, size_t numberOfColumns, T* memoryPtr, bool pinnedMemory, size_t loopUnroll=1, bool subContainer=true);

  const bool sub_container_;
  const bool use_pinned_memory_;
  const ContainerProperties properties_;
  T* memory_ptr_;
};

}
