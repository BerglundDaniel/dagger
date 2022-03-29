#pragma once

#include <stdexcept>

namespace dagger {

/**
 * Exception for Cuda errors.
 */
class CudaException: public std::exception {
public:
  CudaException(const char* errMessage) :
      errMessage(errMessage) {

  }

  const char* what() const throw() {
    return errMessage;
  }

private:
  const char* errMessage;
};

}
