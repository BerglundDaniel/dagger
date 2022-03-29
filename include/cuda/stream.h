#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "exception/cuda_exception.h"
#include "cuda_adapter.h"
#include "device.h"

namespace dagger {

/**
 * This class wraps functionality related to the cuda streams
 */
class Stream {
 public:
  Stream(const Device& device, cudaStream_t* cudaStream, cublasHandle_t* cublasHandle);
  ~Stream();

  const cudaStream_t& cudaStream() const;
  const cublasHandle_t& cublasHandle() const;
  const Device& associatedDevice() const;

  /**
   * Sync the stream
   */
  inline void syncStream() const {
    cudaStreamSynchronize(*cuda_stream_);
  }

  Stream(const Stream&) = delete;
  Stream(Stream&&) = delete;
  Stream& operator=(const Stream&) = delete;
  Stream& operator=(Stream&&) = delete;

 private:
  const Device& device_;
  cudaStream_t* cuda_stream_;
  cublasHandle_t* cublas_handle_;
};

}
