#include "cuda/stream.h"

namespace dagger {

Stream::Stream(const Device& device, cudaStream_t* cudaStream, cublasHandle_t* cublasHandle) :
    device_(device), cuda_stream_(cudaStream), cublas_handle_(cublasHandle) {

}

Stream::~Stream() {
  handleCudaStatus(cudaStreamDestroy(*cuda_stream_), "Failed to destroy CUDA stream:");
  handleCublasStatus(cublasDestroy(*cublas_handle_), "Failed to destroy CuBLAS handle:");

  delete cuda_stream_;
  delete cublas_handle_;
}

const cudaStream_t& Stream::cudaStream() const {
  return *cuda_stream_;
}

const cublasHandle_t& Stream::cublasHandle() const {
  return *cublas_handle_;
}

const Device& Stream::associatedDevice() const {
  return device_;
}

}
