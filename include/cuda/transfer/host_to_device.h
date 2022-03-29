#pragma once

#include <cassert>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "container/device_container_concept.h"
#include "container/device_matrix.h"
#include "container/device_vector.h"
#include "container/host_matrix.h"
#include "container/host_vector.h"
#include "cuda/cuda_adapter.h"
#include "cuda/stream.h"

namespace dagger::host_to_device {
using namespace dagger::container;

/**
 * This namespace contains functions for transfers from the host to the device
 */

/**
 * Copies a matrix on the host to an existing matrix with same size on the device
 */
template<typename T, int loopUnrollHost, int loopUnrollDevice, bool enableTensorCores>
requires deviceContainerReqs<T, enableTensorCores>
void transferMatrix(const Stream& stream, const HostMatrix<T, loopUnrollHost>& hostMatrix,
		    DeviceMatrix<T, loopUnrollDevice, enableTensorCores>& deviceMatrix) {
  const cudaStream_t& cudaStream = stream.cudaStream();
  const int numberOfRows = hostMatrix.numberOfRows();
  const int numberOfColumns = hostMatrix.numberOfColumns();

  assert(numberOfRows == deviceMatrix.numberOfRows());
  assert(numberOfColumns == deviceMatrix.numberOfColumns());

  handleCublasStatus(
      cublasSetMatrixAsync(numberOfRows, numberOfColumns, sizeof(T),
			   hostMatrix.memoryPointer(), hostMatrix.leadingDimension(),
			   deviceMatrix.memoryPointer(), deviceMatrix.leadingDimension(),
			   cudaStream), "Error when transferring matrix from host to device: ");
}

/**
 * Copies a vector on the host to an existing vector with same size on the device
 */
template<typename T, int loopUnrollHost, int loopUnrollDevice, bool enableTensorCores>
requires deviceContainerReqs<T, enableTensorCores>
void transferVector(const Stream& stream, const HostVector<T, loopUnrollHost>& hostVector,
		    DeviceVector<T, loopUnrollDevice, enableTensorCores>& deviceVector) {
  const cudaStream_t& cudaStream = stream.cudaStream();
  const int numberOfRows = hostVector.numberOfRows();

  assert(numberOfRows == deviceVector.numberOfRows());

  handleCublasStatus(
      cublasSetVectorAsync(numberOfRows, sizeof(T),
			   hostVector.memoryPointer(), 1,
			   deviceVector.memoryPointer(), 1,
			   cudaStream), "Error when transferring vector from host to device: ");
}

}
