#include <gtest/gtest.h>
#include <math.h>

#include "container/device_matrix.h"
#include "container/device_matrix.h"

namespace dagger::container {

#ifndef NDEBUG //Only test the assertions if we are in debug version, since assert is not used in releases
TEST(DeviceMatrixDeathTest, Constructor_asserts) {
  ASSERT_DEATH(({DeviceMatrix<float, 1, false>(1, 0);}),"Assertion .* failed");
  ASSERT_DEATH(({DeviceMatrix<float, 1, false>(1, -1);}),"Assertion .* failed");
  ASSERT_DEATH(({DeviceMatrix<int, 1, false>(1, 0);}),"Assertion .* failed");
  ASSERT_DEATH(({DeviceMatrix<int, 1, false>(1, -1);}),"Assertion .* failed");

  ASSERT_DEATH(({DeviceMatrix<float, 1, false>(0, 1);}),"Assertion .* failed");
  ASSERT_DEATH(({DeviceMatrix<float, 1, false>(-1, 1);}),"Assertion .* failed");
  ASSERT_DEATH(({DeviceMatrix<int, 1, false>(0, 1);}),"Assertion .* failed");
  ASSERT_DEATH(({DeviceMatrix<int, 1, false>(-1, 1);}),"Assertion .* failed");

  ASSERT_DEATH(({DeviceMatrix<float, 1, true>(1, 0);}),"Assertion .* failed");
  ASSERT_DEATH(({DeviceMatrix<float, 1, true>(1, -1);}),"Assertion .* failed");
  ASSERT_DEATH(({DeviceMatrix<int, 1, true>(1, 0);}),"Assertion .* failed");
  ASSERT_DEATH(({DeviceMatrix<int, 1, true>(1, -1);}),"Assertion .* failed");

  ASSERT_DEATH(({DeviceMatrix<float, 1, true>(0, 1);}),"Assertion .* failed");
  ASSERT_DEATH(({DeviceMatrix<float, 1, true>(-1, 1);}),"Assertion .* failed");
  ASSERT_DEATH(({DeviceMatrix<int, 1, true>(0, 1);}),"Assertion .* failed");
  ASSERT_DEATH(({DeviceMatrix<int, 1, true>(-1, 1);}),"Assertion .* failed");
}
#endif

TEST(DeviceMatrixTest, initialise_int) {
  const int numberOfRows = 5;
  const int numberOfColumns = 3;
  const int unroll = 1;

  DeviceMatrix<int, unroll, false> matrix(numberOfRows, numberOfColumns);

  EXPECT_EQ(numberOfColumns, matrix.numberOfColumns());
  EXPECT_EQ(numberOfRows, matrix.numberOfRows());
  EXPECT_EQ(numberOfRows, matrix.leadingDimension());
  EXPECT_NE(nullptr, matrix.memoryPointer());
  EXPECT_FALSE(matrix.isSubContainer());
  static_assert(matrix.loopUnrollMultiplier() == unroll);
  static_assert(matrix.leadingDimensionMultiplier() == unroll);

  ContainerProperties prop = matrix.properties();
  EXPECT_EQ(prop.numberOfRows, matrix.numberOfRows());
  EXPECT_EQ(prop.numberOfColumns, matrix.numberOfColumns());
  EXPECT_EQ(prop.leadingDimension, matrix.leadingDimension());
}

TEST(DeviceMatrixTest, initialise_int_tensor) {
  const int numberOfRows = 5;
  const int numberOfColumns = 3;
  const int unroll = 1;
  const int unrollIncTensor = std::lcm(unroll, DAGGER_TENSOR_MULT);;
  const int ld = ceil(((double) numberOfRows) / DAGGER_TENSOR_MULT) * DAGGER_TENSOR_MULT;

  DeviceMatrix<int, unroll, true> matrix(numberOfRows, numberOfColumns);

  EXPECT_EQ(numberOfColumns, matrix.numberOfColumns());
  EXPECT_EQ(numberOfRows, matrix.numberOfRows());
  EXPECT_EQ(ld, matrix.leadingDimension());
  EXPECT_NE(nullptr, matrix.memoryPointer());
  EXPECT_FALSE(matrix.isSubContainer());
  static_assert(matrix.loopUnrollMultiplier() == unroll);
  static_assert(matrix.leadingDimensionMultiplier() == unrollIncTensor);
}

TEST(DeviceMatrixTest, initialise_float) {
  const int numberOfRows = 5;
  const int numberOfColumns = 3;
  const int unroll = 1;

  DeviceMatrix<float, unroll, false> matrix(numberOfRows, numberOfColumns);

  EXPECT_EQ(numberOfColumns, matrix.numberOfColumns());
  EXPECT_EQ(numberOfRows, matrix.numberOfRows());
  EXPECT_EQ(1, matrix.leadingDimensionMultiplier());
  EXPECT_EQ(numberOfRows, matrix.leadingDimension());
  EXPECT_NE(nullptr, matrix.memoryPointer());
  EXPECT_FALSE(matrix.isSubContainer());
  static_assert(matrix.loopUnrollMultiplier() == unroll);
  static_assert(matrix.leadingDimensionMultiplier() == unroll);

  ContainerProperties prop = matrix.properties();
  EXPECT_EQ(prop.numberOfRows, matrix.numberOfRows());
  EXPECT_EQ(prop.numberOfColumns, matrix.numberOfColumns());
  EXPECT_EQ(prop.leadingDimension, matrix.leadingDimension());
}

TEST(DeviceMatrixTest, initialise_float_tensor) {
  const int numberOfRows = 5;
  const int numberOfColumns = 3;
  const int unroll = 1;
  const int unrollIncTensor = std::lcm(unroll, DAGGER_TENSOR_MULT);;
  const int ld = ceil(((double) numberOfRows) / DAGGER_TENSOR_MULT) * DAGGER_TENSOR_MULT;

  DeviceMatrix<float, unroll, true> matrix(numberOfRows, numberOfColumns);

  EXPECT_EQ(numberOfColumns, matrix.numberOfColumns());
  EXPECT_EQ(numberOfRows, matrix.numberOfRows());
  EXPECT_EQ(ld, matrix.leadingDimension());
  EXPECT_NE(nullptr, matrix.memoryPointer());
  EXPECT_FALSE(matrix.isSubContainer());
  static_assert(matrix.loopUnrollMultiplier() == unroll);
  static_assert(matrix.leadingDimensionMultiplier() == unrollIncTensor);
}

TEST(DeviceMatrixTest, multiplier) {
  const int row = 8;
  const int col = 5;
  const int unroll = 3;
  const int ld = ceil(((double) row) / unroll) * unroll;

  DeviceMatrix<float, unroll, false>  matrix(row, col);

  EXPECT_EQ(col, matrix.numberOfColumns());
  EXPECT_EQ(row, matrix.numberOfRows());
  EXPECT_EQ(ld, matrix.leadingDimension());
  EXPECT_NE(nullptr, matrix.memoryPointer());
  EXPECT_FALSE(matrix.isSubContainer());
  static_assert(matrix.loopUnrollMultiplier() == unroll);
  static_assert(matrix.leadingDimensionMultiplier() == unroll);

  ContainerProperties prop = matrix.properties();
  EXPECT_EQ(prop.numberOfRows, matrix.numberOfRows());
  EXPECT_EQ(prop.numberOfColumns, matrix.numberOfColumns());
  EXPECT_EQ(prop.leadingDimension, matrix.leadingDimension());
}

TEST(DeviceMatrixTest, multiplier_tensor) {
  const int row = 8;
  const int col = 5;
  const int unroll = 3;
  const int unrollIncTensor = std::lcm(unroll, DAGGER_TENSOR_MULT);
  const int ld = ceil(((double) row) / unrollIncTensor) * unrollIncTensor;

  DeviceMatrix<float, unroll, true>  matrix(row, col);

  EXPECT_EQ(col, matrix.numberOfColumns());
  EXPECT_EQ(row, matrix.numberOfRows());
  EXPECT_EQ(ld, matrix.leadingDimension());
  EXPECT_NE(nullptr, matrix.memoryPointer());
  EXPECT_FALSE(matrix.isSubContainer());
  static_assert(matrix.loopUnrollMultiplier() == unroll);
  static_assert(matrix.leadingDimensionMultiplier() == unrollIncTensor);

  ContainerProperties prop = matrix.properties();
  EXPECT_EQ(prop.numberOfRows, matrix.numberOfRows());
  EXPECT_EQ(prop.numberOfColumns, matrix.numberOfColumns());
  EXPECT_EQ(prop.leadingDimension, matrix.leadingDimension());
}

}

