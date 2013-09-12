#pragma once
#include <map>
#include <string>
#include <vector>

#include <ncs/sim/DeviceType.h>
#include <ncs/sim/PluginDescription.h>

namespace ncs {

namespace sim {

class DeviceDescription {
public:
  DeviceDescription(bool on_this_machine,
                    double power,
                    DeviceType::Type device_type,
                    int device_index = -1);
  bool isOnThisMachine() const;
  double getPower() const;
  DeviceType::Type getDeviceType() const;
  const std::vector<NeuronPluginDescription*>& getNeuronPlugins() const;
  NeuronPluginDescription* getNeuronPlugin(const std::string& type);
  unsigned int getNeuronPluginIndex(const std::string& type);
  const std::vector<SynapsePluginDescription*>& getSynapsePlugins() const;
  SynapsePluginDescription* getSynapsePlugin(const std::string& type);
  unsigned int getSynapsePluginIndex(const std::string& type);
  int getDeviceIndex() const;
  static std::vector<DeviceDescription*>
    getDevicesOnThisMachine(unsigned int enabled_device_types);
private:
  bool on_this_machine_;
  double power_;
  DeviceType::Type device_type_;
  int device_index_;
  std::vector<NeuronPluginDescription*> neuron_plugins_;
  std::map<std::string, unsigned int> neuron_type_to_plugin_index_;
  std::vector<SynapsePluginDescription*> synapse_plugins_;
  std::map<std::string, unsigned int> synapse_type_to_plugin_index_;
};

} // namespace sim

} // namespace ncs
