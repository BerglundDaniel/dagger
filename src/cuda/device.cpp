#include "cuda/device.h"

namespace dagger {

Device::Device(int deviceNumber) :
    device_number_(deviceNumber){

}

Device::~Device(){

}

bool Device::isActive() const{
  int activeDeviceNumber;
  cudaGetDevice(&activeDeviceNumber);

  if(activeDeviceNumber == device_number_){
    return true;
  }else{
    return false;
  }

}

bool Device::setActiveDevice() const{
  cudaError_t status = cudaSetDevice(device_number_);

  if(status == cudaSuccess){
    return true;
  }else{
    return false;
  }
}

}
