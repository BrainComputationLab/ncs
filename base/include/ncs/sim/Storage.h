#pragma once
#include <ncs/sim/Bit.h>

namespace ncs {

namespace sim {

template<typename T>
struct Storage {
  typedef T type;
  static inline size_t num_elements(size_t count) {
    return count;
  }
};

template<>
struct Storage<Bit> {
  typedef Bit::Word type;
  static inline size_t num_elements(size_t count) {
    return Bit::num_words(count);
  }
};

} // namespace sim

} // namespace ncs
