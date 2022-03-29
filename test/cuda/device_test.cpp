#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "cuda/cuda_adapter.h"
#include "cuda/device.h"
#include "exception/cuda_exception.h"

namespace dagger {

TEST(DeviceTest, set_get_simple) {
  Device device(0);

  device.setActiveDevice();
  ASSERT_TRUE(device.isActive());
}

TEST(DeviceTest, set_get_mult) {
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  if(deviceCount < 2){
    std::cerr << "Skipping test for multiple device since only one device was detected." << std::endl;
    return;
  }

  Device device0(0);
  Device device1(1);

  device1.setActiveDevice();
  ASSERT_TRUE(device1.isActive());
  ASSERT_FALSE(device0.isActive());

  device0.setActiveDevice();
  ASSERT_TRUE(device0.isActive());
  ASSERT_FALSE(device1.isActive());
}

}

