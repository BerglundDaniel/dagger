#pragma once

#include <cassert>
#include <numeric>

namespace dagger::multiplier {

/**
 * Calculate the multiplier to also account for the multiplier needed to be able
 * to use tensor cores.
 */
inline constexpr int calculateMultiplier(bool useTensor, int multiplier) {
  assert(multiplier > 0);
  if (useTensor) {
    //LCM(least common multiplier) is the smallest positive number which is evenly divided by both numbers
    //We use it here because we need a new multiplier that works both for the
    //provided multiplier and the multiplier required for tensor cores to be used in cuda
    return std::lcm(multiplier, DAGGER_TENSOR_MULT);
  } else {
    return multiplier;
  }
}

/**
 * Calculate leading dimension so that it is divisible
 * by the multiplier, and 8 if using tensor cores.
 */
inline constexpr int calculateLeadingDimension(bool useTensor, int length, int multiplier) {
  assert(multiplier > 0);
  assert(length >= 0);
  const int newMultiplier = calculateMultiplier(useTensor, multiplier);

  //Using the fact that we are working with positive integers
  return newMultiplier * ((length + newMultiplier - 1) / newMultiplier);
}

}
