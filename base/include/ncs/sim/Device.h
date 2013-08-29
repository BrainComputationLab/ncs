#pragma once

#include <ncs/sim/DeviceType.h>
#include <ncs/sim/NeuronSimulator.h>

namespace ncs {

namespace sim {

class DeviceBase {
public:
  virtual DeviceType::Type getDeviceType() const = 0;
  virtual unsigned int getNeuronTypeIndex(const std::string& type) const = 0;
private:
};

template<DeviceType::Type MemoryType>
class Device {
public:
  virtual DeviceType::Type getDeviceType() const;
  virtual unsigned int getNeuronTypeIndex(const std::string& type) const;
private:
  std::map<std::string, unsigned int> neuron_type_map_;
  std::vector<NeuronSimulator<MemoryType>*> neuron_simulators_;
};

} // namespace sim

} // namespace ncs

#include <ncs/sim/Device.hpp>
