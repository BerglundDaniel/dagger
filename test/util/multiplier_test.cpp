#include <gtest/gtest.h>
#include <math.h>
#include <numeric>

#include "util/multiplier.h"

namespace dagger::multiplier  {

#ifndef NDEBUG //Only test the assertions if we are in debug version, since assert is not used in releases
TEST(MultiplierDeathTest, calculate_length) {
  ASSERT_DEATH(({calculateLeadingDimension(false, 1, 0);}),"Assertion .* failed");
  ASSERT_DEATH(({calculateLeadingDimension(false, -1, 1);}),"Assertion .* failed");
}

TEST(MultiplierDeathTest, calculate_multiplier) {
  ASSERT_DEATH(({calculateMultiplier(false, 0);}),"Assertion .* failed");
  ASSERT_DEATH(({calculateMultiplier(false, -1);}),"Assertion .* failed");
}
#endif

TEST(MultiplierTest, calculate_length_constexpr) {
  const int multiplier = 3;
  const int length = 10;
  constexpr int res = ceil(((double) length) / (double) multiplier) * multiplier;
  constexpr int ld = calculateLeadingDimension(false, length, multiplier);
  static_assert(res == ld);
}

TEST(MultiplierTest, calculate_length_constexpr_tensor) {
  const int multiplier = 3;
  const int length = 10;
  constexpr int multiplierIncTensor = std::lcm(DAGGER_TENSOR_MULT,3);
  constexpr int res = ceil(((double) length) / (double) multiplierIncTensor) * multiplierIncTensor;
  constexpr int ld = calculateLeadingDimension(true, length, multiplier);
  static_assert(res == ld);
}

TEST(MultiplierTest, calculate_multiplier_constexpr) {
  const int multiplier = 3;
  constexpr int ld = calculateMultiplier(false, multiplier);
  static_assert(multiplier == ld);
}

TEST(MultiplierTest, calculate_multiplier_constexpr_tensor) {
  const int multiplier = 3;
  constexpr int res = std::lcm(multiplier, DAGGER_TENSOR_MULT);
  constexpr int ld = calculateMultiplier(true, multiplier);
  static_assert(res == ld);
}

TEST(MultiplierTest, calculate_length) {
  const int multiplier = 3;
  int res;
  int ld;

  for(int length = 8; length<30; ++length) {
    res = ceil(((double) length) / (double) multiplier) * multiplier;
    ld = calculateLeadingDimension(false, length, multiplier);
    EXPECT_EQ(res, ld);
  }
}

TEST(MultiplierTest, calculate_length_tensor) {
  const int multiplier = 3;
  constexpr int multiplierIncTensor = std::lcm(DAGGER_TENSOR_MULT,3);
  int res;
  int ld;

  for(int length = 8; length<30; ++length) {
    res = ceil(((double) length) / (double) multiplierIncTensor) * multiplierIncTensor;
    ld = calculateLeadingDimension(true, length, multiplier);
    EXPECT_EQ(res, ld);
  }
}

TEST(MultiplierTest, calculate_multiplier) {
  int res;

  for(int multiplier = 1; multiplier<30; ++multiplier) {
    res = calculateMultiplier(false, multiplier);
    EXPECT_EQ(res, multiplier);
  }
}

TEST(MultiplierTest, calculate_multiplier_tensor) {
  int res;

  for(int multiplier = 1; multiplier<30; ++multiplier) {
    res = calculateMultiplier(true, multiplier);
    EXPECT_EQ(res, std::lcm(multiplier, DAGGER_TENSOR_MULT));
  }
}

}
