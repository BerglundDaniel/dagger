#pragma once

#include <stdexcept>

namespace dagger {

/**
 * Exception when the state of a class is incorrect
 */
class InvalidState: public std::exception {
public:
  InvalidState(const char* errMessage) :
      errMessage(errMessage) {

  }

  const char* what() const throw() {
    return errMessage;
  }

private:
  const char* errMessage;
};

}
