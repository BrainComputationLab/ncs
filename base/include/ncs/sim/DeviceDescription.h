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
  DeviceDescription(bool on_this_machine, double power, DeviceType::Type device_type);
  bool isOnThisMachine() const;
  double getPower() const;
  DeviceType::Type getDeviceType() const;
  const std::vector<NeuronPluginDescription*>& getNeuronPlugins() const;
  NeuronPluginDescription* getNeuronPlugin(const std::string& type);
  unsigned int getNeuronPluginIndex(const std::string& type);
  static std::vector<DeviceDescription*>
    getDevicesOnThisMachine(unsigned int enabled_device_types);
private:
  bool on_this_machine_;
  double power_;
  DeviceType::Type device_type_;
  std::vector<NeuronPluginDescription*> neuron_plugins_;
  std::map<std::string, unsigned int> neuron_type_to_plugin_index_;
};

} // namespace sim

} // namespace ncs
