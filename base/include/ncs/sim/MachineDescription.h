#pragma once
#include <vector>

#include <ncs/sim/DeviceDescription.h>

namespace ncs {

namespace sim {

class MachineDescription {
public:
  MachineDescription(const std::vector<DeviceDescription*>& devices);
  const std::vector<DeviceDescription*>& getDevices() const;
  static MachineDescription* getThisMachine(unsigned int enabled_device_types);
private:
  std::vector<DeviceDescription*> devices_;
};

} // namespace sim

} // namespace ncs
