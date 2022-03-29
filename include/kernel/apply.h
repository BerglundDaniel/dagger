#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cassert>

#include "cuda/cuda_adapter.h"
#include "container/device_vector.h"
#include "container/device_matrix.h"
#include "container/container_properties.h"
#include "internal/apply.cuh"

namespace dagger::kernel {

extern template void internal::applyInternal<int,5>(); //TODO macro this

template<typename T, int  loopUnroll, bool enableTensorCores>
inline void apply(int numBlocks, int threadsPerBlock,
		  const container::DeviceVector<T, loopUnroll, enableTensorCores>& vectorIn,
		   container::DeviceVector<T, loopUnroll, enableTensorCores>& vectorOut){
  assert(vectorIn.leadingDimension() == vectorOut.leadingDimension());
  assert(numBlocks > 0 && threadsPerBlock > 0);
  internal::applyInternal<T, loopUnroll>(numBlocks, threadsPerBlock, vectorIn.properties(),
					 vectorIn.memoryPointer(), vectorOut.memoryPointer());
}

}
