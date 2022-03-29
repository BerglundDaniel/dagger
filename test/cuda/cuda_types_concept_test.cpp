#include <gtest/gtest.h>

#include "cuda/cuda_types_concept.h"

namespace dagger {

TEST(CudaTypesConcept, cuda_types_valid_types) {
  static_assert(cudaTypes<float>);
  static_assert(cudaTypes<short>);
  static_assert(cudaTypes<int>);
  static_assert(cudaTypes<double>);
}

TEST(CudaTypesConcept, cuda_types_invalid_types) {
  static_assert(!cudaTypes<std::string>);
}

TEST(CudaTypesConcept, tensor_types_valid_types) {
  static_assert(tensorTypes<short>);
  static_assert(tensorTypes<float>);
}

TEST(CudaTypesConcept, tensor_types_invalid_types) {
  static_assert(!tensorTypes<std::string>);
  static_assert(!tensorTypes<long long>);
  static_assert(!tensorTypes<long long>);
  static_assert(!tensorTypes<double>);
  static_assert(!tensorTypes<long int>);
}

}
