#pragma once

#include <ncs/sim/DeviceType.h>
#include <ncs/sim/FactoryMap.h>
#include <ncs/sim/NeuronSimulator.h>
#include <ncs/sim/SynapseSimulator.h>

namespace ncs {

namespace sim {

class DeviceBase {
public:
  virtual DeviceType::Type getDeviceType() const = 0;
  virtual int getNeuronTypeIndex(const std::string& type) const = 0;
  virtual bool initialize(DeviceDescription* description,
                          FactoryMap<NeuronSimulator>* neuron_plugins,
                          FactoryMap<SynapseSimulator>* synapse_plugins) = 0;
private:
};

template<DeviceType::Type MemoryType>
class Device : public DeviceBase {
public:
  virtual DeviceType::Type getDeviceType() const;
  virtual int getNeuronTypeIndex(const std::string& type) const;
  virtual bool initialize(DeviceDescription* description,
                          FactoryMap<NeuronSimulator>* neuron_plugins,
                          FactoryMap<SynapseSimulator>* synapse_plugins);
private:
  std::map<std::string, int> neuron_type_map_;
  std::vector<NeuronSimulator<MemoryType>*> neuron_simulators_;
};

} // namespace sim

} // namespace ncs

#include <ncs/sim/Device.hpp>
