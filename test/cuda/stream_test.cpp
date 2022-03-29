#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "cuda/cuda_adapter.h"
#include "cuda/stream.h"
#include "exception/cuda_exception.h"
#include "mocks/device_mock.h"

namespace dagger {

TEST(StreamTest, getters) {
  cudaStream_t* cudaStream = new cudaStream_t();
  cublasHandle_t* cublasHandle = new cublasHandle_t();

  handleCublasStatus(cublasCreate(cublasHandle), "Failed to create new cublas handle:");
  handleCudaStatus(cudaStreamCreate(cudaStream), "Failed to create new cuda stream:");
  handleCublasStatus(cublasSetStream(*cublasHandle, *cudaStream), "Failed to set cuda stream for cublas handle:");

  DeviceMock deviceMock;

  Stream stream(deviceMock, cudaStream, cublasHandle);

  EXPECT_EQ(cudaStream, &stream.cudaStream());
  EXPECT_EQ(cublasHandle, &stream.cublasHandle());
  EXPECT_EQ(&deviceMock, &stream.associatedDevice());
}

}

