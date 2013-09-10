namespace ncs {

namespace sim {

template<DeviceType::Type MemoryType>
DeviceType::Type Device<MemoryType>::getDeviceType() const {
  return MemoryType;
}

template<DeviceType::Type MemoryType>
int Device<MemoryType>::getNeuronTypeIndex(const std::string& type) const {
  auto result = neuron_type_map_.find(type);
  if (result == neuron_type_map_.end()) {
    std::cerr << "No neuron of type " << type << " found on this device." <<
      std::endl;
    return -1;
  }
  return result->second;
}

template<DeviceType::Type MemoryType>
bool Device<MemoryType>::
initialize(DeviceDescription* description,
           FactoryMap<NeuronSimulator>* neuron_plugins,
           FactoryMap<SynapseSimulator>* synapse_plugins) {
  for (auto plugin_description : description->getNeuronPlugins()) {
    const std::string& type = plugin_description->getType();
    auto generator = neuron_plugins->getProducer<MemoryType>(type);
    if (!generator) {
      std::cerr << "Neuron plugin for type " << type << " for device type " <<
        DeviceType::as_string(MemoryType) << " was not found." << std::endl;
      return false;
    }
  }
  return true;
}

} // namespace sim

} // namespace ncs
