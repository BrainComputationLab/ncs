#include <ncs/sim/DeviceType.h>

namespace ncs {

namespace sim {

const std::string& DeviceType::as_string(Type type) {
  static std::string unknown = "unknown";
  static std::string cuda = "CUDA";
  static std::string cpu = "CPU";
  static std::string cl = "CL";
  static std::string invalid = "invalid";
  switch(type) {
    case UNKNOWN:
      return unknown;
      break;
    case CUDA:
      return cuda;
      break;
    case CPU:
      return cpu;
      break;
    case CL:
      return cl;
      break;
    default:
      return invalid;
      break;
  }
}

} // namespace sim

} // namespace ncs
