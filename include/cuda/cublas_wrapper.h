#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>
#include <sstream>
#include <string>

#include "../container/device_matrix.h"
#include "../container/device_vector.h"
#include "../exception/cublas_exception.h"
#include "../exception/cuda_exception.h"
#include "cuda_adapter.cu"
#include "stream.h"

namespace Dagger::CuBLAS {

using namespace Dagger::Container;

  /**
   * Wrapper for CUBLAS
   */

  /**
   * Copies from vectorFrom to vectorTo element wise
   */
  void copyVector(const Stream& stream, const DeviceVector& vectorFrom, DeviceVector& vectorTo);

  void matrixVectorMultiply(const Stream& stream, const DeviceMatrix& matrix, const DeviceVector& vector,
      DeviceVector& result);

  void matrixTransVectorMultiply(const Stream& stream, const DeviceMatrix& matrix, const DeviceVector& vector,
      DeviceVector& result);

  void matrixTransMatrixMultiply(const Stream& stream, const DeviceMatrix& matrix1, const DeviceMatrix& matrix2,
      DeviceMatrix& result);

  void sumResultToHost(const Stream& stream, const DeviceVector& vector, const DeviceVector& oneVector,
      PRECISION& sumHost);

  //TODO template this
  void absoluteSumToHost(const Stream& stream, const DeviceVector& vector, PRECISION& sumHost);

}
