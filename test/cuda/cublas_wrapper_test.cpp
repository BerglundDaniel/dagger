#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdlib.h>

#include "container/host_matrix.h"
#include "container/host_vector.h"
#include "container/device_matrix.h"
#include "container/device_vector.h"
#include "cuda/cublas_wrapper.h"
#include "cuda/cuda_adapter.h"
#include "cuda/device.h"
#include "cuda/stream.h"
#include "cuda/stream_factory.h"
#include "cuda/transfer/device_to_host.h"
#include "cuda/transfer/host_to_device.h"
#include "exception/cublas_exception.h"
#include "exception/cuda_exception.h"

namespace Dagger::CuBLAS {

  //TODO make more tests
  class CublasWrapperTest: public ::testing::Test {
  protected:
    CublasWrapperTest();
    virtual ~CublasWrapperTest();
    virtual void SetUp();
    virtual void TearDown();

    Device device;
    StreamFactory streamFactory;
    Stream* stream;
  };

  CublasWrapperTest::CublasWrapperTest() :
      device(0), streamFactory(), stream(streamFactory.constructStream(device)) {

  }

  CublasWrapperTest::~CublasWrapperTest() {
    delete stream;
  }

  void CublasWrapperTest::SetUp() {

  }

  void CublasWrapperTest::TearDown() {

  }

  TEST_F(CublasWrapperTest, matrixTransMatrixMultiply) {
    const int numberOfRows = 5;
    const int numberOfColumns = 3;
    const int numberOfRows2 = 5;
    const int numberOfColumns2 = 4;
    PinnedHostMatrix matrixT(numberOfRows, numberOfColumns);
    PinnedHostMatrix matrix2(numberOfRows2, numberOfColumns2);

    matrixT(0, 0) = 1;
    matrixT(1, 0) = 2;
    matrixT(2, 0) = 3;
    matrixT(3, 0) = 4;
    matrixT(4, 0) = 5;

    matrixT(0, 1) = 10;
    matrixT(1, 1) = 20;
    matrixT(2, 1) = 30;
    matrixT(3, 1) = 40;
    matrixT(4, 1) = 50;

    matrixT(0, 2) = 1.1;
    matrixT(1, 2) = 2.2;
    matrixT(2, 2) = 3.3;
    matrixT(3, 2) = 4.4;
    matrixT(4, 2) = 5.5;

    for(int i = 0; i < numberOfRows2; ++i){
      matrix2(i, 0) = 6;
    }

    for(int i = 0; i < numberOfRows2; ++i){
      matrix2(i, 1) = 7;
    }

    for(int i = 0; i < numberOfRows2; ++i){
      matrix2(i, 2) = 8;
    }

    for(int i = 0; i < numberOfRows2; ++i){
      matrix2(i, 3) = 9;
    }

    DeviceMatrix* matrixTDevice = HostToDevice::transferMatrix(*stream, matrixT);
    DeviceMatrix* matrix2Device = HostToDevice::transferMatrix(*stream, matrix2);
    DeviceMatrix* resultDevice = new DeviceMatrix(numberOfColumns, numberOfColumns2);

    matrixTransMatrixMultiply(*stream, *matrixTDevice, *matrix2Device, *resultDevice);
    stream->syncStream();
    handleCudaStatus(cudaGetLastError(), "Error with matrixTransMatrixMultiply in matrixTransMatrixMultiply: ");

    PinnedHostMatrix* resultHost = DeviceToHost::transferMatrix(*stream, *resultDevice);
    stream->syncStream();
    handleCudaStatus(cudaGetLastError(), "Error with transfer in matrixTransMatrixMultiply: ");

    EXPECT_EQ(90, (*resultHost)(0, 0));
    EXPECT_EQ(105, (*resultHost)(0, 1));
    EXPECT_EQ(120, (*resultHost)(0, 2));
    EXPECT_EQ(135, (*resultHost)(0, 3));

    EXPECT_EQ(900, (*resultHost)(1, 0));
    EXPECT_EQ(1050, (*resultHost)(1, 1));
    EXPECT_EQ(1200, (*resultHost)(1, 2));
    EXPECT_EQ(1350, (*resultHost)(1, 3));

    EXPECT_EQ(99, (*resultHost)(2, 0));
    EXPECT_EQ(115.5, (*resultHost)(2, 1));
    EXPECT_EQ(132, (*resultHost)(2, 2));
    EXPECT_EQ(148.5, (*resultHost)(2, 3));

    delete matrixTDevice;
    delete matrix2Device;
    delete resultDevice;
    delete resultHost;
  }

}
