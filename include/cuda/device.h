#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "container/device_vector.h"
#include "exception/cuda_exception.h"
#include "exception/invalid_state.h"
#include "cuda_adapter.h"


namespace dagger {

/**
  Wraps functions regarding the devices and controls which device is currently active
 */
class Device {
public:
  Device(int deviceNumber);
  ~Device();

  /**
    Checks if the device is the current active device
  */
  bool isActive() const;

  /**
    Set this device as the active device. Returns false if failed
  */
  bool setActiveDevice() const;

  Device(const Device&) = delete;
  Device(Device&&) = delete;
  Device& operator=(const Device&) = delete;
  Device& operator=(Device&&) = delete;

private:
  const int device_number_;
};

}
