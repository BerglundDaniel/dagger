#pragma once

#include <stdexcept>

namespace dagger {

/**
 * Exception for Cublas errors.
 */

class CublasException: public std::exception {
public:
  CublasException(const char* errMessage) :
      errMessage(errMessage) {

  }

  const char* what() const throw() {
    return errMessage;
  }

private:
  const char* errMessage;
};

}
