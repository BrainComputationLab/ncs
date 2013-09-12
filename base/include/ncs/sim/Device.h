#pragma once

#include <ncs/sim/DataBuffer.h>
#include <ncs/sim/DeviceDescription.h>
#include <ncs/sim/DeviceType.h>
#include <ncs/sim/FactoryMap.h>
#include <ncs/sim/NeuronSimulator.h>
#include <ncs/sim/PluginDescription.h>
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
  bool initializeNeurons_(DeviceDescription* description,
                          FactoryMap<NeuronSimulator>* neuron_plugins);
  bool initializeNeuronSimulator_(NeuronSimulator<MemoryType>* simulator,
                                  NeuronPluginDescription* description);
  bool initializeNeuronVoltages_();

  bool initializeSynapses_(DeviceDescription* description,
                           FactoryMap<SynapseSimulator>* synapse_plugins);
  bool initializeSynapseSimulator_(SynapseSimulator<MemoryType>* simulator,
                                  SynapsePluginDescription* description);

  std::map<std::string, int> neuron_type_map_;
  std::vector<NeuronSimulator<MemoryType>*> neuron_simulators_;
  std::vector<size_t> neuron_device_id_offsets_;
  size_t neuron_device_vector_size_;

  std::vector<SynapseSimulator<MemoryType>*> synapse_simulators_;
};

} // namespace sim

} // namespace ncs

#include <ncs/sim/Device.hpp>
