#pragma once
#include <cstddef>

namespace ncs {

namespace sim {

class DataType {
public:
  enum Type {
    Bit,
    Float,
    Integer,
    Unknown
  };
  static size_t num_bytes(std::size_t count, Type t);
private:
};

} // namespace sim

} // namespace ncs
