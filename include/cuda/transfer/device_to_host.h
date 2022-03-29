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

namespace dagger::device_to_host {
using namespace dagger::container;

/**
 * This namespace contains functions for transfers from the device to the host
 */

/**
 * Copies a matrix on the device to an existing matrix with same size on the host
 */
template<typename T, int loopUnrollHost, int loopUnrollDevice, bool enableTensorCores>
requires deviceContainerReqs<T, enableTensorCores>
void transferMatrix(const Stream& stream, const DeviceMatrix<T, loopUnrollDevice, enableTensorCores>& deviceMatrix,
		    HostMatrix<T, loopUnrollHost>& hostMatrix) {
  const cudaStream_t& cudaStream = stream.cudaStream();
  const int numberOfRows = deviceMatrix.numberOfRows();
  const int numberOfColumns = deviceMatrix.numberOfColumns();

  assert(numberOfRows == hostMatrix.numberOfRows());
  assert(numberOfColumns == hostMatrix.numberOfColumns());

  handleCublasStatus(
      cublasGetMatrixAsync(numberOfRows, numberOfColumns, sizeof(T),
			   deviceMatrix.memoryPointer(), deviceMatrix.leadingDimension(),
			   hostMatrix.memoryPointer(), hostMatrix.leadingDimension(),
			   cudaStream), "Error when transferring matrix from device to host: ");
}

/**
 * Copies a vector on the device to an existing vector with same size on the host
 */
template<typename T, int loopUnrollHost, int loopUnrollDevice, bool enableTensorCores>
requires deviceContainerReqs<T, enableTensorCores>
void transferVector(const Stream& stream, const DeviceVector<T, loopUnrollDevice, enableTensorCores>& deviceVector,
		    HostVector<T, loopUnrollHost>& hostVector) {
  const cudaStream_t& cudaStream = stream.cudaStream();
  const int numberOfRows = deviceVector.numberOfRows();

  assert(numberOfRows == hostVector.numberOfRows());

  handleCublasStatus(
      cublasGetVectorAsync(numberOfRows, sizeof(T),
			   deviceVector.memoryPointer(), 1,
			   hostVector.memoryPointer(), 1,
			   cudaStream), "Error when transferring vector from device to host: ");
}

}
