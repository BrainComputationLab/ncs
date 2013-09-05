#include <iostream>

#include <ncs/sim/MachineDescription.h>

namespace ncs {

namespace sim {

MachineDescription::
MachineDescription(const std::vector<DeviceDescription*>& devices)
  : devices_(devices) {
}

const std::vector<DeviceDescription*>&
MachineDescription::getDevices() const {
  return devices_;
}

MachineDescription*
MachineDescription::getThisMachine(unsigned int enabled_device_types) {
  std::vector<DeviceDescription*> devices =
    DeviceDescription::getDevicesOnThisMachine(enabled_device_types);
  return new MachineDescription(devices);
}

} // namespace sim

} // namespace ncs
