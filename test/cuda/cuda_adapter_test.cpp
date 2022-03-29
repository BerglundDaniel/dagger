#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <string>
#include <sstream>

#include "cuda/cuda_adapter.h"
#include "exception/cublas_exception.h"
#include "exception/cuda_exception.h"

namespace dagger {

TEST(CudaAdapterTest, cublas_error_string) {
  EXPECT_EQ("CUBLAS_STATUS_SUCCESS", cublasGetErrorString(CUBLAS_STATUS_SUCCESS));
  EXPECT_EQ("CUBLAS_STATUS_NOT_INITIALIZED", cublasGetErrorString(CUBLAS_STATUS_NOT_INITIALIZED));
  EXPECT_EQ("CUBLAS_STATUS_ALLOC_FAILED", cublasGetErrorString(CUBLAS_STATUS_ALLOC_FAILED));
  EXPECT_EQ("CUBLAS_STATUS_INVALID_VALUE", cublasGetErrorString(CUBLAS_STATUS_INVALID_VALUE));
  EXPECT_EQ("CUBLAS_STATUS_ARCH_MISMATCH", cublasGetErrorString(CUBLAS_STATUS_ARCH_MISMATCH));
  EXPECT_EQ("CUBLAS_STATUS_MAPPING_ERROR", cublasGetErrorString(CUBLAS_STATUS_MAPPING_ERROR));
  EXPECT_EQ("CUBLAS_STATUS_EXECUTION_FAILED", cublasGetErrorString(CUBLAS_STATUS_EXECUTION_FAILED));
  EXPECT_EQ("CUBLAS_STATUS_INTERNAL_ERROR", cublasGetErrorString(CUBLAS_STATUS_INTERNAL_ERROR));
}

TEST(CudaAdapterTest, handle_cuda_status) {
  handleCudaStatus(cudaSuccess, "test");
  EXPECT_THROW(handleCudaStatus(cudaErrorMissingConfiguration, "test"), CudaException);
  EXPECT_THROW(handleCudaStatus(cudaErrorNoDevice, "test"), CudaException);
  EXPECT_THROW(handleCudaStatus(cudaErrorIncompatibleDriverContext, "test"), CudaException);
}

TEST(CudaAdapterTest, handle_cublas_status) {
  handleCublasStatus(CUBLAS_STATUS_SUCCESS, "test");
  EXPECT_THROW(handleCublasStatus(CUBLAS_STATUS_NOT_INITIALIZED, "test"), CublasException);
  EXPECT_THROW(handleCublasStatus(CUBLAS_STATUS_ALLOC_FAILED, "test"), CublasException);
  EXPECT_THROW(handleCublasStatus(CUBLAS_STATUS_INVALID_VALUE, "test"), CublasException);
  EXPECT_THROW(handleCublasStatus(CUBLAS_STATUS_ARCH_MISMATCH, "test"), CublasException);
  EXPECT_THROW(handleCublasStatus(CUBLAS_STATUS_MAPPING_ERROR, "test"), CublasException);
  EXPECT_THROW(handleCublasStatus(CUBLAS_STATUS_EXECUTION_FAILED, "test"), CublasException);
  EXPECT_THROW(handleCublasStatus(CUBLAS_STATUS_INTERNAL_ERROR, "test"), CublasException);
}

}

