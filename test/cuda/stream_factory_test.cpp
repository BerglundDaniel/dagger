#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "cuda/cuda_adapter.h"
#include "cuda/stream.h"
#include "cuda/stream_factory.h"
#include "exception/cuda_exception.h"
#include "mocks/device_mock.h"

using testing::Return;

namespace dagger {

TEST(StreamFactoryTest, construct_stream) {
  StreamFactory streamFactory;
  DeviceMock deviceMock;

  EXPECT_CALL(deviceMock, isActive()).Times(1).WillRepeatedly(Return(true));

  Stream* stream = streamFactory.constructStream(deviceMock);

  ASSERT_EQ(&deviceMock, &stream->associatedDevice());

  delete stream;
}

}

