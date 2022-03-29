#ifndef DAGGER_APPLY_H_
#define DAGGER_APPLY_H_

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "container/container_properties.h"

//This file is compiled by nvcc so it has to be C++11

namespace dagger {
namespace kernal {
namespace internal {

#ifdef __CUDACC__
/*
 * The kernel implementation
 */
template<typename T, int loopUnroll>
__global__
inline void applyKernel(const container::ContainerProperties prop, const T* vecIn, T* vecOut) {

}

#endif

/*
 * The function that calls the actual kernel
 */
template<typename T, int loopUnroll>
void applyInternal(int numBlocks, int threadsPerBlock, const container::ContainerProperties prop, const T* vecIn, T* vecOut) {
#ifdef __CUDACC__
  applyKernel<T, loopUnroll><<<numBlocks, threadsPerBlock>>>(prop, vecIn, vecOut);
#endif
}

} /* internal */
} /* kernel */
} /* dagger */

#endif /* DAGGER_APPLY_H_ */
