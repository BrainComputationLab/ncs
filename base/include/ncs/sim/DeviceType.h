#pragma once

#include <string>

namespace ncs {

namespace sim {

class DeviceType {
public:
  enum Type {
    UNKNOWN = 0,
    CPU = 1,
    CUDA = 2,
    CL = 4
  };
  static const std::string& as_string(Type type);
};

} // namespace sim

} // namespace ncs
