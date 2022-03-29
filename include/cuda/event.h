#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "exception/cuda_exception.h"

#include "cuda_adapter.h"
#include "stream.h"

namespace dagger {

  /**
   * This class wraps CUDA events
   */
  class Event {
  public:
    Event(const Stream& stream);
    virtual ~Event();

    float operator-(Event& otherEvent);

  #ifndef __CUDACC__
    Event(const Event&) = delete;
    Event(Event&&) = delete;
    Event& operator=(const Event&) = delete;
    Event& operator=(Event&&) = delete;
  #endif

    cudaEvent_t cuda_event_;

  private:
    const Stream& stream_;
  };

}
