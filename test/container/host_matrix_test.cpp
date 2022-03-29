#include <gtest/gtest.h>
#include <math.h>

#include "container/container_properties.h"
#include "container/host_matrix.h"

namespace dagger::container {

#ifndef NDEBUG //Only test the assertions if we are in debug version, since assert is not used in releases
TEST(HostMatrixDeathTest, Constructor_asserts) {
  ASSERT_DEATH(({HostMatrix<float, 1>(1, 0);}),"Assertion .* failed");
  ASSERT_DEATH(({HostMatrix<float, 1>(1, -1);}),"Assertion .* failed");
  ASSERT_DEATH(({HostMatrix<int, 1>(1, 0);}),"Assertion .* failed");
  ASSERT_DEATH(({HostMatrix<int, 1>(1, -1);}),"Assertion .* failed");

  ASSERT_DEATH(({HostMatrix<float, 1>(0, 1);}),"Assertion .* failed");
  ASSERT_DEATH(({HostMatrix<float, 1>(-1, 1);}),"Assertion .* failed");
  ASSERT_DEATH(({HostMatrix<int, 1>(0, 1);}),"Assertion .* failed");
  ASSERT_DEATH(({HostMatrix<int, 1>(-1, 1);}),"Assertion .* failed");

  ASSERT_DEATH(({HostMatrix<float, 1>(1, 0, false);}),"Assertion .* failed");
  ASSERT_DEATH(({HostMatrix<float, 1>(1, -1, false);}),"Assertion .* failed");
  ASSERT_DEATH(({HostMatrix<int, 1>(1, 0, false);}),"Assertion .* failed");
  ASSERT_DEATH(({HostMatrix<int, 1>(1, -1, false);}),"Assertion .* failed");

  ASSERT_DEATH(({HostMatrix<float, 1>(0, 1, false);}),"Assertion .* failed");
  ASSERT_DEATH(({HostMatrix<float, 1>(-1, 1, false);}),"Assertion .* failed");
  ASSERT_DEATH(({HostMatrix<int, 1>(0, 1, false);}),"Assertion .* failed");
  ASSERT_DEATH(({HostMatrix<int, 1>(-1, 1, false);}),"Assertion .* failed");
}
#endif

TEST(HostMatrixTest, init) {
  const int rows = 5;
  const int cols = 3;
  const int unroll = 1;

  HostMatrix<float, unroll> pinnedMatrix(rows, cols);
  HostMatrix<float, unroll> regularMatrix(rows, cols, false);

  EXPECT_EQ(cols, pinnedMatrix.numberOfColumns());
  EXPECT_EQ(rows, pinnedMatrix.numberOfRows());
  EXPECT_EQ(rows, pinnedMatrix.leadingDimension());
  EXPECT_FALSE(pinnedMatrix.isSubContainer());
  static_assert(pinnedMatrix.loopUnrollMultiplier() == unroll);
  static_assert(pinnedMatrix.leadingDimensionMultiplier() == unroll);

  EXPECT_EQ(cols, regularMatrix.numberOfColumns());
  EXPECT_EQ(rows, regularMatrix.numberOfRows());
  EXPECT_EQ(rows, regularMatrix.leadingDimension());
  EXPECT_FALSE(regularMatrix.isSubContainer());
  static_assert(regularMatrix.loopUnrollMultiplier() == unroll);
  static_assert(regularMatrix.leadingDimensionMultiplier() == unroll);

  ContainerProperties pinnedProp = pinnedMatrix.properties();
  EXPECT_EQ(pinnedProp.numberOfRows, pinnedMatrix.numberOfRows());
  EXPECT_EQ(pinnedProp.numberOfColumns, pinnedMatrix.numberOfColumns());
  EXPECT_EQ(pinnedProp.leadingDimension, pinnedMatrix.leadingDimension());

  ContainerProperties regProp = regularMatrix.properties();
  EXPECT_EQ(regProp.numberOfRows, regularMatrix.numberOfRows());
  EXPECT_EQ(regProp.numberOfColumns, regularMatrix.numberOfColumns());
  EXPECT_EQ(regProp.leadingDimension, regularMatrix.leadingDimension());
}

TEST(HostMatrixTest, access_operator) {
  const int rows = 5;
  const int cols = 3;
  HostMatrix<float, 1> pinnedMatrix(rows, cols);
  HostMatrix<float, 1> regularMatrix(rows, cols, false);

  float a = 5;
  float b = 3.2;

  pinnedMatrix(0, 1) = a;
  pinnedMatrix(3, 2) = b;

  EXPECT_EQ(a, pinnedMatrix(0, 1));
  EXPECT_EQ(b, pinnedMatrix(3, 2));

  regularMatrix(0, 1) = a;
  regularMatrix(3, 2) = b;

  EXPECT_EQ(a, regularMatrix(0, 1));
  EXPECT_EQ(b, regularMatrix(3, 2));
}


TEST(HostMatrixTest, multiplier) {
  const int row = 8;
  const int col = 5;
  const int unroll = 3;
  HostMatrix<float, unroll> hostMatrix(row, col, true);

  const int ld = ceil(((double) row) / unroll) * unroll;

  EXPECT_EQ(col, hostMatrix.numberOfColumns());
  EXPECT_EQ(row, hostMatrix.numberOfRows());
  EXPECT_EQ(ld, hostMatrix.leadingDimension());
  static_assert(hostMatrix.loopUnrollMultiplier() == unroll);
  static_assert(hostMatrix.leadingDimensionMultiplier() == unroll);
}

TEST(HostMatrixTest, int_test) {
  const size_t rows = 7;
  const size_t cols = 4;
  const int unroll = 3;
  const int ld = ceil(((double) rows) / unroll) * unroll;

  HostMatrix<int, unroll> pinnedMatrix(rows, cols);
  HostMatrix<int, unroll> regularMatrix(rows, cols, false);

  EXPECT_EQ(cols, pinnedMatrix.numberOfColumns());
  EXPECT_EQ(rows, pinnedMatrix.numberOfRows());
  EXPECT_EQ(ld, pinnedMatrix.leadingDimension());
  static_assert(pinnedMatrix.loopUnrollMultiplier() == unroll);
  static_assert(pinnedMatrix.leadingDimensionMultiplier() == unroll);

  EXPECT_EQ(cols, regularMatrix.numberOfColumns());
  EXPECT_EQ(rows, regularMatrix.numberOfRows());
  EXPECT_EQ(ld, regularMatrix.leadingDimension());
  static_assert(regularMatrix.loopUnrollMultiplier() == unroll);
  static_assert(regularMatrix.leadingDimensionMultiplier() == unroll);

  int a = 10;
  int b = 50;

  pinnedMatrix(0, 1) = a;
  pinnedMatrix(3, 2) = b;

  EXPECT_EQ(a, pinnedMatrix(0, 1));
  EXPECT_EQ(b, pinnedMatrix(3, 2));

  regularMatrix(0, 1) = a;
  regularMatrix(3, 2) = b;

  EXPECT_EQ(a, regularMatrix(0, 1));
  EXPECT_EQ(b, regularMatrix(3, 2));
}

}

