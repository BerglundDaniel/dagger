#pragma once

#include "cuda/cuda_types_concept.h"

namespace dagger::container {

template <class T, bool enableTensorCores>
concept deviceContainerReqs = (cudaTypes<T> && !enableTensorCores ) || (tensorTypes<T> && enableTensorCores);

}
