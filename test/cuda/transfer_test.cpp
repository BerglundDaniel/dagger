#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "container/device_matrix.h"
#include "container/device_vector.h"
#include "cuda/cuda_adapter.h"
#include "cuda/device.h"
#include "cuda/stream.h"
#include "cuda/stream_factory.h"
#include "cuda/transfer/device_to_host.h"
#include "cuda/transfer/host_to_device.h"
#include "exception/cublas_exception.h"
#include "exception/cuda_exception.h"

namespace dagger {
using namespace dagger::container;
using namespace dagger::device_to_host;
using namespace dagger::host_to_device;

/**
 * Assumes that the container classes are working properly
 */
class TransferTest: public ::testing::Test {
 protected:
  TransferTest();
  virtual ~TransferTest();
  virtual void SetUp();
  virtual void TearDown();

  Device device;
  StreamFactory streamFactory;
  Stream* stream;
};

TransferTest::TransferTest() :
    device(0), streamFactory(), stream(streamFactory.constructStream(device)) {

}

TransferTest::~TransferTest() {
  delete stream;
}

void TransferTest::SetUp() {

}

void TransferTest::TearDown() {

}

TEST_F(TransferTest, transfer_vector_float) {
  const int numberOfRows = 5;

  HostVector<float, 1> hostVectorFrom(numberOfRows);
  HostVector<float, 1> hostVectorTo(numberOfRows);
  DeviceVector<float, 1, false> deviceVector(numberOfRows);

  for(int i = 0; i < numberOfRows; ++i){
    hostVectorFrom(i) = i + 10.1f;
  }

  transferVector(*stream, hostVectorFrom, deviceVector);
  transferVector(*stream, deviceVector, hostVectorTo);
  stream->syncStream();
  handleCudaStatus(cudaGetLastError(), "Error when transferring vector in test: ");

  for(int i = 0; i < numberOfRows; ++i){
    EXPECT_EQ(i + 10.1f, hostVectorFrom(i));
    EXPECT_EQ(i + 10.1f, hostVectorTo(i));
  }
}

TEST_F(TransferTest, transfer_vector_float_tensor) {
  const int numberOfRows = 5;

  HostVector<float, 1> hostVectorFrom(numberOfRows);
  HostVector<float, 1> hostVectorTo(numberOfRows);
  DeviceVector<float, 1, true> deviceVector(numberOfRows);

  for(int i = 0; i < numberOfRows; ++i){
    hostVectorFrom(i) = i + 10.2f;
  }

  transferVector(*stream, hostVectorFrom, deviceVector);
  transferVector(*stream, deviceVector, hostVectorTo);
  stream->syncStream();
  handleCudaStatus(cudaGetLastError(), "Error when transferring vector in test: ");

  for(int i = 0; i < numberOfRows; ++i){
    EXPECT_EQ(i + 10.2f, hostVectorFrom(i));
    EXPECT_EQ(i + 10.2f, hostVectorTo(i));
  }
}

TEST_F(TransferTest, transfer_vector_int) {
  const int numberOfRows = 5;

  HostVector<int, 1> hostVectorFrom(numberOfRows);
  HostVector<int, 1> hostVectorTo(numberOfRows);
  DeviceVector<int, 1, false> deviceVector(numberOfRows);

  for(int i = 0; i < numberOfRows; ++i){
    hostVectorFrom(i) = i + 10;
  }

  transferVector(*stream, hostVectorFrom, deviceVector);
  transferVector(*stream, deviceVector, hostVectorTo);
  stream->syncStream();
  handleCudaStatus(cudaGetLastError(), "Error when transferring vector in test: ");

  for(int i = 0; i < numberOfRows; ++i){
    EXPECT_EQ(i + 10, hostVectorFrom(i));
    EXPECT_EQ(i + 10, hostVectorTo(i));
  }
}

TEST_F(TransferTest, transfer_matrix_int) {
  const int numberOfRows = 5;
  const int numberOfColumns = 3;

  HostMatrix<int, 1> hostMatrixFrom(numberOfRows, numberOfColumns);
  HostMatrix<int, 1> hostMatrixTo(numberOfRows, numberOfColumns);
  DeviceMatrix<int, 1, false> deviceMatrix(numberOfRows, numberOfColumns);

  for(int j = 0; j < numberOfColumns; ++j){
    for(int i = 0; i < numberOfRows; ++i){
      hostMatrixFrom(i, j) = i + (10 * j);
    }
  }

  transferMatrix(*stream, hostMatrixFrom, deviceMatrix);
  transferMatrix(*stream, deviceMatrix, hostMatrixTo);
  stream->syncStream();
  handleCudaStatus(cudaGetLastError(), "Error when transferring matrix in test: ");

  for(int j = 0; j < numberOfColumns; ++j){
    for(int i = 0; i < numberOfRows; ++i){
      EXPECT_EQ(i + (10 * j), hostMatrixFrom(i, j));
      EXPECT_EQ(i + (10 * j), hostMatrixTo(i, j));
    }
  }
}

TEST_F(TransferTest, transfer_vector_unroll) {
  const int numberOfRows = 5;

  HostVector<float, 3> hostVectorFrom(numberOfRows);
  HostVector<float, 2> hostVectorTo(numberOfRows);
  DeviceVector<float, 4, false> deviceVector(numberOfRows);

  for(int i = 0; i < numberOfRows; ++i){
    hostVectorFrom(i) = i + 10.1f;
  }

  transferVector(*stream, hostVectorFrom, deviceVector);
  transferVector(*stream, deviceVector, hostVectorTo);
  stream->syncStream();
  handleCudaStatus(cudaGetLastError(), "Error when transferring vector in test: ");

  for(int i = 0; i < numberOfRows; ++i){
    EXPECT_EQ(i + 10.1f, hostVectorFrom(i));
    EXPECT_EQ(i + 10.1f, hostVectorTo(i));
  }
}

}

