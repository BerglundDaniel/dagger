#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "cuda/cuda_adapter.h"
#include "cuda/event.h"
#include "cuda/stream.h"
#include "cuda/stream_factory.h"
#include "exception/cuda_exception.h"
#include "mocks/device_mock.h"

using testing::Ge;
using testing::Le;

namespace dagger {

TEST(EventTest, event_difference) {
  const double e = 1e-2;
  StreamFactory streamFactory;

  Device device(0);
  device.setActiveDevice();
  ASSERT_TRUE(device.isActive());

  Stream* stream = streamFactory.constructStream(device);

  Event before(*stream);

  //TODO a kernel here that waits a while

  Event after(*stream);

  stream->syncStream();

  float diff = after - before;

  const float realDiff = 1;
  float l = realDiff - e;
  float h = realDiff + e;
  EXPECT_THAT(diff, Ge(l));
  EXPECT_THAT(diff, Le(h));

  delete stream;
}

}
