#include <gtest/gtest.h>
#include <math.h>

#include "container/container_properties.h"
#include "container/device_vector.h"

namespace dagger::container {

#ifndef NDEBUG //Only test the assertions if we are in debug version, since assert is not used in releases
TEST(DeviceVectorDeathTest, Constructor_asserts) {
  ASSERT_DEATH(({DeviceVector<float, 1, false>(0);}),"Assertion .* failed");
  ASSERT_DEATH(({DeviceVector<float, 1, false>(-1);}),"Assertion .* failed");
  ASSERT_DEATH(({DeviceVector<int, 1, false>(0);}),"Assertion .* failed");
  ASSERT_DEATH(({DeviceVector<int, 1, false>(-1);}),"Assertion .* failed");

  ASSERT_DEATH(({DeviceVector<float, 1, true>(0);}),"Assertion .* failed");
  ASSERT_DEATH(({DeviceVector<float, 1, true>(-1);}),"Assertion .* failed");
  ASSERT_DEATH(({DeviceVector<int, 1, true>(0);}),"Assertion .* failed");
  ASSERT_DEATH(({DeviceVector<int, 1, true>(-1);}),"Assertion .* failed");
}
#endif

TEST(DeviceVectorTest, initialise_int) {
  const int size = 5;
  const int unroll = 1;

  DeviceVector<int, unroll, false> vector(size);

  EXPECT_EQ(1, vector.numberOfColumns());
  EXPECT_EQ(size, vector.numberOfRows());
  EXPECT_EQ(size, vector.leadingDimension());
  EXPECT_NE(nullptr, vector.memoryPointer());
  EXPECT_FALSE(vector.isSubContainer());
  static_assert(vector.loopUnrollMultiplier() == unroll);
  static_assert(vector.leadingDimensionMultiplier() == unroll);

  ContainerProperties prop = vector.properties();
  EXPECT_EQ(prop.numberOfRows, vector.numberOfRows());
  EXPECT_EQ(prop.numberOfColumns, vector.numberOfColumns());
  EXPECT_EQ(prop.leadingDimension, vector.leadingDimension());
}

TEST(DeviceVectorTest, initialise_int_tensor) {
  const int size = 5;
  const int unroll = 1;
  const int unrollIncTensor = std::lcm(unroll, DAGGER_TENSOR_MULT);
  const int ld = ceil(((double) size) / unrollIncTensor) * unrollIncTensor;

  DeviceVector<int, unroll, true> vector(size);

  EXPECT_EQ(1, vector.numberOfColumns());
  EXPECT_EQ(size, vector.numberOfRows());
  EXPECT_EQ(ld, vector.leadingDimension());
  EXPECT_NE(nullptr, vector.memoryPointer());
  EXPECT_FALSE(vector.isSubContainer());
  static_assert(vector.loopUnrollMultiplier() == unroll);
  static_assert(vector.leadingDimensionMultiplier() == unrollIncTensor);
}

TEST(DeviceVectorTest, initialise_float) {
  const int size = 5;
  const int unroll = 1;

  DeviceVector<float, unroll, false> vector(size);

  EXPECT_EQ(1, vector.numberOfColumns());
  EXPECT_EQ(size, vector.numberOfRows());
  EXPECT_EQ(size, vector.leadingDimension());
  EXPECT_NE(nullptr, vector.memoryPointer());
  EXPECT_FALSE(vector.isSubContainer());
  static_assert(vector.loopUnrollMultiplier() == unroll);
  static_assert(vector.leadingDimensionMultiplier() == unroll);

  ContainerProperties prop = vector.properties();
  EXPECT_EQ(prop.numberOfRows, vector.numberOfRows());
  EXPECT_EQ(prop.numberOfColumns, vector.numberOfColumns());
  EXPECT_EQ(prop.leadingDimension, vector.leadingDimension());
}

TEST(DeviceVectorTest, initialise_float_tensor) {
  const int size = 5;
  const int unroll = 1;
  const int unrollIncTensor = std::lcm(unroll, DAGGER_TENSOR_MULT);
  const int ld = ceil(((double) size) / unrollIncTensor) * unrollIncTensor;

  DeviceVector<float, unroll, true> vector(size);

  EXPECT_EQ(1, vector.numberOfColumns());
  EXPECT_EQ(size, vector.numberOfRows());
  EXPECT_EQ(ld, vector.leadingDimension());
  EXPECT_NE(nullptr, vector.memoryPointer());
  EXPECT_FALSE(vector.isSubContainer());
  static_assert(vector.loopUnrollMultiplier() == unroll);
  static_assert(vector.leadingDimensionMultiplier() == unrollIncTensor);
}

TEST(DeviceVectorTest, multiplier) {
  const int size = 7;
  const int unroll = 3;
  const int ld = ceil(((double) size) / unroll) * unroll;

  DeviceVector<float, unroll, false> vector(size);

  EXPECT_EQ(1, vector.numberOfColumns());
  EXPECT_EQ(size, vector.numberOfRows());
  EXPECT_EQ(ld, vector.leadingDimension());
  static_assert(vector.loopUnrollMultiplier() == unroll);
  static_assert(vector.leadingDimensionMultiplier() == unroll);

  ContainerProperties prop = vector.properties();
  EXPECT_EQ(prop.numberOfRows, vector.numberOfRows());
  EXPECT_EQ(prop.numberOfColumns, vector.numberOfColumns());
  EXPECT_EQ(prop.leadingDimension, vector.leadingDimension());
}

TEST(DeviceVectorTest, multiplier_tensor) {
  const int size = 7;
  const int unroll = 3;
  const int unrollIncTensor = std::lcm(unroll, DAGGER_TENSOR_MULT);
  const int ld = ceil(((double) size) / unrollIncTensor) * unrollIncTensor;

  DeviceVector<float, unroll, true> vector(size);

  EXPECT_EQ(1, vector.numberOfColumns());
  EXPECT_EQ(size, vector.numberOfRows());
  EXPECT_EQ(ld, vector.leadingDimension());
  static_assert(vector.loopUnrollMultiplier() == unroll);
  static_assert(vector.leadingDimensionMultiplier() == unrollIncTensor);

  ContainerProperties prop = vector.properties();
  EXPECT_EQ(prop.numberOfRows, vector.numberOfRows());
  EXPECT_EQ(prop.numberOfColumns, vector.numberOfColumns());
  EXPECT_EQ(prop.leadingDimension, vector.leadingDimension());
}

}

