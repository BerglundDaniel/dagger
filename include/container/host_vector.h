#pragma once

#include <cstdlib>
#include <cassert>

#include "cuda/cuda_adapter.h"
#include "cuda/cuda_types_concept.h"
#include "util/multiplier.h"
#include "container_properties.h"

namespace dagger::container {

/**
 * Vector with memory stored on the host.
 * Pinned memory prevents paging. It is faster to transfer and required for async tranfers
 * Allocated amount of memory is a multiplier of lengthMultiplier.
 * The class is the owner of the memory unless it is a subcontainer.
 * Column major
 */
template <typename T, int loopUnroll>
requires cudaTypes<T> && (loopUnroll > 0)
class HostVector {
 public:
  explicit HostVector(int numberOfRows, bool usePinnedMemory=true)
      : properties_{
          .numberOfRows = numberOfRows,
          .numberOfColumns = 1,
          .leadingDimension = multiplier::calculateLeadingDimension(false, numberOfRows, loopUnroll),},
        use_pinned_memory_(usePinnedMemory), sub_container_(false), memory_ptr_(nullptr) {
    assert(numberOfRows > 0);
    if(use_pinned_memory_){
      allocateHostPinnedMemory<T>((void**) &(memory_ptr_), properties_.leadingDimension);
    } else {
      memory_ptr_ = (T*) malloc(sizeof(T) * properties_.leadingDimension);
    }
  }

  ~HostVector() {
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
    return 1;
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

  inline T& operator()(int pos) {
    return *(memory_ptr_ + pos);
  }

  inline const T operator()(int pos) const {
    return *(memory_ptr_ + pos);
  }

  inline T* memoryPointer() {
    return memory_ptr_;
  }

  inline const T* memoryPointer() const {
    return memory_ptr_;
  }

  HostVector(const HostVector&) = delete;
  HostVector(HostVector&&) = delete;
  HostVector& operator=(const HostVector&) = delete;
  HostVector& operator=(HostVector&&) = delete;

 private:
  //explicit HostVector(size_t numberOfRealRows, size_t numberOfRows, bool pinnedMemory=true, size_t loopUnroll=1);
  //explicit HostVector(size_t numberOfRealRows, size_t numberOfRows, T* memoryPtr, bool pinnedMemory, size_t loopUnroll=1, bool subContainer=true);

  const bool sub_container_;
  const bool use_pinned_memory_;
  const ContainerProperties properties_;
  T* memory_ptr_;
};

}
