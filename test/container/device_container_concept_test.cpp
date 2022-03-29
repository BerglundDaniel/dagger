#include <gtest/gtest.h>
#include <string>
#include <sstream>

#include "container/device_container_concept.h"

namespace dagger::container {

TEST(CudaContainerConcept, valid_types) {
  static_assert(deviceContainerReqs<float, false>);
  static_assert(deviceContainerReqs<double, false>);
  static_assert(deviceContainerReqs<short, false>);
  static_assert(deviceContainerReqs<int, false>);
  static_assert(deviceContainerReqs<long, false>);
  static_assert(deviceContainerReqs<long long, false>);

  static_assert(deviceContainerReqs<float, true>);
  static_assert(deviceContainerReqs<short, true>);
  static_assert(deviceContainerReqs<int, true>);
}

TEST(CudaContainerConcept, invalid_types) {
  static_assert(!deviceContainerReqs<std::string, false>);
  static_assert(!deviceContainerReqs<std::string, true>);
  static_assert(!deviceContainerReqs<long long, true>);
  static_assert(!deviceContainerReqs<double, true>);
}

}
