#pragma once

#include "device_vector.h"
#include "device_matrix.h"
#include "container_properties.h"
#include "device_container_concept.h"

namespace dagger::container {

//TODO grid factory? eller inte?
//TODO transfer?

/**
 * X
 */
template <class T, bool enableTensorCores>
requires deviceContainerReqs<T, bool enableTensorCores>
class Grid {
public:
  explicit Grid(int numberOfRows, int lengthMultiplier=1) :
    mainMatrix(asdf)
  {

  }

  virtual ~Grid() {

  }

  DeviceMatrix<T, enableTensorCores> main() {
    return mainMatrix;
  }

private:
  DeviceMatrix<T, enableTensorCores> mainMatrix;
};

}
