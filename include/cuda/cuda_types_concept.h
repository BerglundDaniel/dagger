#pragma once

#include <limits.h>

namespace dagger {

template <class T>
concept cudaTypes = (std::integral<T> || std::floating_point<T>) && std::numeric_limits<T>::digits <= 64;

template <class T>
  concept tensorTypes = (std::integral<T> || std::floating_point<T>) && std::numeric_limits<T>::digits <= 32;
}
