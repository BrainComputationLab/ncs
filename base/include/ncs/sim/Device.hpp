#include <ncs/sim/Bit.h>

namespace ncs {

namespace sim {

template<DeviceType::Type MType>
DeviceType::Type Device<MType>::getDeviceType() const {
  return MType;
}

template<DeviceType::Type MType>
int Device<MType>::getNeuronTypeIndex(const std::string& type) const {
  auto result = neuron_type_map_.find(type);
  if (result == neuron_type_map_.end()) {
    std::cerr << "No neuron of type " << type << " found on this device." <<
      std::endl;
    return -1;
  }
  return result->second;
}

template<DeviceType::Type MType>
bool Device<MType>::
initialize(DeviceDescription* description,
           FactoryMap<NeuronSimulator>* neuron_plugins,
           FactoryMap<SynapseSimulator>* synapse_plugins) {
  std::clog << "Initializing neurons..." << std::endl;
  if (!initializeNeurons_(description, neuron_plugins)) {
    std::cerr << "Failed to initialize neurons." << std::endl;
    return false;
  }

  std::clog << "Initializing NeuronSimulatorUpdater..." << std::endl;
  if (!initializeNeuronUpdater_()) {
    std::cerr << "Failed to initialize NeuronSimulatorUpdater." << std::endl;
    return false;
  }

  std::clog << "Initializing vector exchangers..." << std::endl;
  if (!initializeVectorExchangers_()) {
    std::cerr << "Failed to initialize vector exchangers." << std::endl;
    return false;
  }

  std::clog << "Initializing synapses..." << std::endl;
  if (!initializeSynapses_(description, synapse_plugins)) {
    std::cerr << "Failed to initialize synapses." << std::endl;
    return false;
  }
  return true;
}

template<DeviceType::Type MType>
bool Device<MType>::
initializeNeurons_(DeviceDescription* description,
                   FactoryMap<NeuronSimulator>* neuron_plugins) {
  size_t neuron_device_id_offset = 0;
  for (auto plugin_description : description->getNeuronPlugins()) {
    const std::string& type = plugin_description->getType();
    auto generator = neuron_plugins->getProducer<MType>(type);
    if (!generator) {
      std::cerr << "Neuron plugin for type " << type << " for device type " <<
        DeviceType::as_string(MType) << " was not found." << std::endl;
      return false;
    }
    NeuronSimulator<MType>* simulator = generator();
    if (!initializeNeuronSimulator_(simulator, plugin_description)) {
      std::cerr << "Failed to initialize neuron plugin." << std::endl;
      delete simulator;
      return false;
    }
    neuron_simulators_.push_back(simulator);
    neuron_device_id_offsets_.push_back(neuron_device_id_offset);
    neuron_device_id_offset += Bit::pad(neuron_device_id_offset);
  }
  neuron_device_vector_size_ = neuron_device_id_offset;
  return true;
}

template<DeviceType::Type MType>
bool Device<MType>::
initializeNeuronSimulator_(NeuronSimulator<MType>* simulator,
                           NeuronPluginDescription* description) {
  for (auto neuron : description->getNeurons()) {
    if (!simulator->addNeuron(neuron)) {
      std::cerr << "Failed to add a neuron to the simulator." << std::endl;
      return false;
    }
  }
  if (!simulator->initialize()) {
    std::cerr << "Failed to initialize a neuron simulator." << std::endl;
    return false;
  }
  return true;
}

template<DeviceType::Type MType>
bool Device<MType>::initializeNeuronUpdater_() {
  neuron_simulator_updater_ = new NeuronSimulatorUpdater<MType>();
  return true;
}

template<DeviceType::Type MType>
bool Device<MType>::initializeVectorExchangers_() {
  fire_vector_extractor_ = new DeviceVectorExtractor<MType>();
  if (!fire_vector_extractor_->init(neuron_simulator_updater_)) {
    std::cerr << "Failed to initialize DeviceVectorExtractor." << std::endl;
    return false;
  }
  return true;
}

template<DeviceType::Type MType>
bool Device<MType>::
initializeSynapses_(DeviceDescription* description,
                    FactoryMap<SynapseSimulator>* synapse_plugins) {
  for (auto plugin_description : description->getSynapsePlugins()) {
    const std::string& type = plugin_description->getType();
    auto generator = synapse_plugins->getProducer<MType>(type);
    if (!generator) {
      std::cerr << "Synapse plugin for type " << type << " for device type " <<
        DeviceType::as_string(MType) << " was not found." << std::endl;
      return false;
    }
    SynapseSimulator<MType>* simulator = generator();
    if (!initializeSynapseSimulator_(simulator, plugin_description)) {
      std::cerr << "Failed to initialize synapse plugin." << std::endl;
      delete simulator;
      return false;
    }
    synapse_simulators_.push_back(simulator);
  }
  return true;
}

template<DeviceType::Type MType>
bool Device<MType>::
initializeSynapseSimulator_(SynapseSimulator<MType>* simulator,
                            SynapsePluginDescription* description) {
  for (auto synapse : description->getSynapses()) {
    if (!simulator->addSynapse(synapse)) {
      std::cerr << "Failed to add a synapse to the simulator." << std::endl;
      return false;
    }
  }
  if (!simulator->initialize()) {
    std::cerr << "Failed to initialize a synapse simulator." << std::endl;
    return false;
  }
  return true;
}

} // namespace sim

} // namespace ncs
