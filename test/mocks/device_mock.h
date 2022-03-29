#pragma once

#include <gmock/gmock.h>

#include "container/device_vector.h"
#include "cuda/device.h"

namespace dagger {

class DeviceMock: public Device {
public:
  DeviceMock() :
  Device(0){

  }

  virtual ~DeviceMock() {

  }

  MOCK_CONST_METHOD0(isActive, bool());
  MOCK_CONST_METHOD0(setActiveDevice, bool());
};

}
