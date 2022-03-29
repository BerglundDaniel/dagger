#pragma once

#include <cassert>

#include "cuda/cuda_adapter.h"
#include "util/multiplier.h"
#include "container_properties.h"
#include "device_container_concept.h"

namespace dagger::container {

/**
 * Vector with memory stored on the device.
 * Allocated amount of memory is a multiplier of lengthMultiplier and 8 if using tensor cores.
 * The class is the owner of the memory unless it is a subcontainer.
 * Column major
 */
template <typename T, int loopUnroll, bool enableTensorCores>
requires deviceContainerReqs<T, enableTensorCores> && (loopUnroll > 0)
class DeviceVector {
 public:
  explicit DeviceVector(int numberOfRows)
      : properties_{
          .numberOfRows = numberOfRows,
          .numberOfColumns = 1,
          .leadingDimension =  multiplier::calculateLeadingDimension(enableTensorCores, numberOfRows, loopUnroll)},
      sub_container_(false), memory_ptr_(nullptr) {
    assert(numberOfRows > 0);
    allocateDeviceMemory<T>((void**) &memory_ptr_, properties_.leadingDimension);
  }

  ~DeviceVector() {
    if(!sub_container_){
      freeDeviceMemory(memory_ptr_);
    }
  }

  inline constexpr int loopUnrollMultiplier() {
    return loopUnroll;
  }

  inline constexpr int leadingDimensionMultiplier() const {
    return multiplier::calculateMultiplier(enableTensorCores, loopUnroll);
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
  inline T* memoryPointer() {
    return memory_ptr_;
  }

  inline const T* memoryPointer() const{
    return memory_ptr_;
  }

  DeviceVector(const DeviceVector&) = delete;
  DeviceVector(DeviceVector&&) = delete;
  DeviceVector& operator=(const DeviceVector&) = delete;
  DeviceVector& operator=(DeviceVector&&) = delete;

 private:
  const ContainerProperties properties_;
  const bool sub_container_;
  T* memory_ptr_;
};

}
