#include <ncs/sim/Location.h>

namespace ncs {

namespace sim {

Location::Location()
  : machine(-1),
    device(-1),
    plugin(-1) {
}

Location::Location(int m, int d, int p)
  : machine(m),
    device(d),
    plugin(p) {
}

bool Location::operator==(const Location& r) const {
  return machine == r.machine &&
    device == r.device &&
    plugin == r.plugin;
}

bool Location::operator<(const Location& r) const {
  if (machine < r.machine) {
    return true;
  }
  if (machine > r.machine) {
    return false;
  }
  // Machines are equal
  if (device < r.device) {
    return true;
  }
  if (device > r.device) {
    return false;
  }
  // Devices are equal
  if (plugin < r.plugin) {
    return true;
  }
  return false;
}

} // namespace sim

} // namespace ncs
