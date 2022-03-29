#include "cuda/event.h"

namespace dagger {

Event::Event(const Stream& stream) :
    stream_(stream) {
  handleCudaStatus(cudaEventCreate(&cuda_event_, cudaEventDefault), "Failed to create CUDA event");
  handleCudaStatus(cudaEventRecord(cuda_event_, stream_.cudaStream()), "Failed to record CUDA event on stream.");
}

Event::~Event() {

}

float Event::operator-(Event& otherEvent) {
  float timeElapsed;
  handleCudaStatus(cudaEventElapsedTime(&timeElapsed, otherEvent.cuda_event_, cuda_event_),
      "Failed to get elapsed time between CUDA events.");
  return timeElapsed;
}

}
