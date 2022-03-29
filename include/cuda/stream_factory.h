#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "exception/cuda_exception.h"

#include "cuda_adapter.h"
#include "device.h"
#include "stream.h"

namespace dagger {

class StreamFactory {
public:
  StreamFactory();
  virtual ~StreamFactory();

  Stream* constructStream(const Device& device) const;
};

}
