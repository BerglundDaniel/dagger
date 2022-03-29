#include <gtest/gtest.h>
#include <math.h>

#include "container/container_properties.h"
#include "container/host_vector.h"

namespace dagger::container {

#ifndef NDEBUG //Only test the assertions if we are in debug version, since assert is not used in releases
TEST(HostVectorDeathTest, Constructor_asserts) {
  ASSERT_DEATH(({HostVector<float, 1>(0);}),"Assertion .* failed");
  ASSERT_DEATH(({HostVector<float, 1>(-1);}),"Assertion .* failed");
  ASSERT_DEATH(({HostVector<int, 1>(0);}),"Assertion .* failed");
  ASSERT_DEATH(({HostVector<int, 1>(-1);}),"Assertion .* failed");

  ASSERT_DEATH(({HostVector<float, 1>(0, false);}),"Assertion .* failed");
  ASSERT_DEATH(({HostVector<float, 1>(-1, false);}),"Assertion .* failed");
  ASSERT_DEATH(({HostVector<int, 1>(0, false);}),"Assertion .* failed");
  ASSERT_DEATH(({HostVector<int, 1>(-1, false);}),"Assertion .* failed");
}
#endif

TEST(HostVectorTest, init) {
  const int size = 5;
  const int unroll = 1;

  HostVector<float, unroll> pinnedVector(size);
  HostVector<float, unroll> regularVector(size, false);

  EXPECT_TRUE(pinnedVector.isPinnedMemory());
  EXPECT_FALSE(regularVector.isPinnedMemory());

  EXPECT_EQ(1, pinnedVector.numberOfColumns());
  EXPECT_EQ(size, pinnedVector.numberOfRows());
  EXPECT_EQ(size, pinnedVector.leadingDimension());
  EXPECT_FALSE(pinnedVector.isSubContainer());
  static_assert(pinnedVector.loopUnrollMultiplier() == unroll);
  static_assert(pinnedVector.leadingDimensionMultiplier() == unroll);

  EXPECT_EQ(1, regularVector.numberOfColumns());
  EXPECT_EQ(size, regularVector.numberOfRows());
  EXPECT_EQ(size, regularVector.leadingDimension());
  EXPECT_FALSE(regularVector.isSubContainer());
  static_assert(regularVector.loopUnrollMultiplier() == unroll);
  static_assert(regularVector.leadingDimensionMultiplier() == unroll);

  ContainerProperties pinnedProp = pinnedVector.properties();
  EXPECT_EQ(pinnedProp.numberOfRows, pinnedVector.numberOfRows());
  EXPECT_EQ(pinnedProp.numberOfColumns, pinnedVector.numberOfColumns());
  EXPECT_EQ(pinnedProp.leadingDimension, pinnedVector.leadingDimension());

  ContainerProperties regProp = regularVector.properties();
  EXPECT_EQ(regProp.numberOfRows, regularVector.numberOfRows());
  EXPECT_EQ(regProp.numberOfColumns, regularVector.numberOfColumns());
  EXPECT_EQ(regProp.leadingDimension, regularVector.leadingDimension());
}

TEST(HostVectorTest, access_operator) {
  const int size = 5;
  HostVector<float, 1> pinnedVector(size);
  HostVector<float, 1> regularVector(size, false);

  float a = 5;
  float b = 3.2;

  pinnedVector(0) = a;
  pinnedVector(3) = b;
  regularVector(0) = a;
  regularVector(3) = b;

  EXPECT_EQ(a, pinnedVector(0));
  EXPECT_EQ(b, pinnedVector(3));
  EXPECT_EQ(a, regularVector(0));
  EXPECT_EQ(b, regularVector(3));
}

TEST(HostVectorTest, multiplier) {
  const int size = 8;
  const int unroll = 3;
  const int ld = ceil(((double) size) / unroll) * unroll;

  HostVector<float, unroll> pinnedVector(size, true);
  EXPECT_FALSE(pinnedVector.isSubContainer());

  HostVector<float, unroll> regularVector(size, false);
  EXPECT_FALSE(regularVector.isSubContainer());

  static_assert(pinnedVector.loopUnrollMultiplier() == unroll);
  static_assert(pinnedVector.leadingDimensionMultiplier() == unroll);
  EXPECT_EQ(1, pinnedVector.numberOfColumns());
  EXPECT_EQ(size, pinnedVector.numberOfRows());
  EXPECT_EQ(ld, pinnedVector.leadingDimension());

  static_assert(regularVector.loopUnrollMultiplier() == unroll);
  static_assert(regularVector.leadingDimensionMultiplier() == unroll);
  EXPECT_EQ(1, regularVector.numberOfColumns());
  EXPECT_EQ(size, regularVector.numberOfRows());
  EXPECT_EQ(ld, regularVector.leadingDimension());
}

TEST(HostVectorTest, int_test) {
  const int size = 8;
  const int unroll = 3;
  const int ld = ceil(((double) size) / unroll) * unroll;

  HostVector<int, unroll> regularVector(size, false);
  EXPECT_FALSE(regularVector.isSubContainer());

  HostVector<int, unroll> pinnedVector(size, true);
  EXPECT_FALSE(regularVector.isSubContainer());

  EXPECT_TRUE(pinnedVector.isPinnedMemory());
  EXPECT_FALSE(regularVector.isPinnedMemory());

  EXPECT_EQ(1, pinnedVector.numberOfColumns());
  EXPECT_EQ(size, pinnedVector.numberOfRows());
  EXPECT_EQ(ld, pinnedVector.leadingDimension());
  EXPECT_FALSE(pinnedVector.isSubContainer());

  EXPECT_EQ(1, regularVector.numberOfColumns());
  EXPECT_EQ(size, regularVector.numberOfRows());
  EXPECT_EQ(ld, regularVector.leadingDimension());
  EXPECT_FALSE(regularVector.isSubContainer());

  int a = 2;
  int b = 1;

  pinnedVector(0) = a;
  pinnedVector(3) = b;
  regularVector(0) = a;
  regularVector(3) = b;

  EXPECT_EQ(a, pinnedVector(0));
  EXPECT_EQ(b, pinnedVector(3));
  EXPECT_EQ(a, regularVector(0));
  EXPECT_EQ(b, regularVector(3));
}

}

