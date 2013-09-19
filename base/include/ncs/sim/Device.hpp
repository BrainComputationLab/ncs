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
           FactoryMap<SynapseSimulator>* synapse_plugins,
           MachineVectorExchanger* machine_vector_exchanger,
           size_t global_neuron_vector_size,
           SpecificPublisher<StepSignal>* signal_publisher) {
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
  if (!initializeVectorExchangers_(machine_vector_exchanger,
                                   global_neuron_vector_size)) {
    std::cerr << "Failed to initialize vector exchangers." << std::endl;
    return false;
  }

  std::clog << "Initializing synapses..." << std::endl;
  if (!initializeSynapses_(description, synapse_plugins)) {
    std::cerr << "Failed to initialize synapses." << std::endl;
    return false;
  }

  std::clog << "Initializing FireTable..." << std::endl;
  if (!initializeFireTable_()) {
    std::cerr << "Failed to initialize fire table." << std::endl;
    return false;
  }

  std::clog << "Initializing FireTableUpdater..." << std::endl;
  if (!initializeFireTableUpdater_(description)) {
    std::cerr << "Failed to initialize FireTableUpdater." << std::endl;
    return false;
  }

  std::clog << "Initializing InputUpdater..." << std::endl;
  if (!initializeInputUpdater_(signal_publisher)) {
    std::cerr << "Failed to initialize InputUpdater." << std::endl;
    return false;
  }

  return true;
}

template<DeviceType::Type MType>
bool Device<MType>::threadInit() {
  return true;
}

template<DeviceType::Type MType>
bool Device<MType>::threadDestroy() {
  return true;
}

template<DeviceType::Type MType>
bool Device<MType>::start() {
  std::clog << "Starting InputUpdater..." << std::endl;
  auto input_updater_function = [=]() {
    std::clog << "Stepping" << std::endl;
    while (input_updater_->step()) {
      std::clog << "step" << std::endl;
    }
    std::clog << "Shutting down InputUpdater..." << std::endl;
    delete input_updater_;
  };
  input_updater_thread_ = std::thread(input_updater_function);
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
bool Device<MType>::
initializeVectorExchangers_(MachineVectorExchanger* machine_exchanger,
                            size_t global_neuron_vector_size) {
  fire_vector_extractor_ = new DeviceVectorExtractor<MType>();
  if (!fire_vector_extractor_->init(neuron_simulator_updater_)) {
    std::cerr << "Failed to initialize DeviceVectorExtractor." << std::endl;
    return false;
  }

  global_vector_injector_ =
    new GlobalVectorInjector<MType>(global_neuron_vector_size,
                                    Constants::num_buffers);
  if (!global_vector_injector_->init(machine_exchanger)) {
    std::cerr << "Failed to initialize GlobalVectorInjector." << std::endl;
    return false;
  }

  return true;
}

template<DeviceType::Type MType>
bool Device<MType>::
initializeSynapses_(DeviceDescription* description,
                    FactoryMap<SynapseSimulator>* synapse_plugins) {
  min_synaptic_delay_ = std::numeric_limits<unsigned int>::max();
  max_synaptic_delay_ = std::numeric_limits<unsigned int>::min();
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
  if (description->getSynapsePlugins().empty()) {
    min_synaptic_delay_ = 1;
    max_synaptic_delay_ = 1;
  }
  if (0 == min_synaptic_delay_) {
    std::cerr << "Synapses cannot have zero delay." << std::endl;
    return false;
  }
  device_synaptic_vector_size_ = 0;
  size_t synapse_device_id_offset = 0;
  for (auto plugin_description : description->getSynapsePlugins()) {
    synapse_device_id_offsets_.push_back(synapse_device_id_offset);
    synapse_device_id_offset +=
      Bit::pad(plugin_description->getSynapses().size());
  }
  device_synaptic_vector_size_ = synapse_device_id_offset;
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
    min_synaptic_delay_ = std::min(min_synaptic_delay_, synapse->delay);
    max_synaptic_delay_ = std::max(max_synaptic_delay_, synapse->delay);
  }
  if (!simulator->initialize()) {
    std::cerr << "Failed to initialize a synapse simulator." << std::endl;
    return false;
  }
  return true;
}

template<DeviceType::Type MType>
bool Device<MType>::
initializeFireTable_() {
  fire_table_ = new FireTable<MType>(device_synaptic_vector_size_,
                                     min_synaptic_delay_,
                                     max_synaptic_delay_);
  if (!fire_table_->init()) {
    std::cerr << "Failed to initialize FireTable." << std::endl;
    return false;
  }
  return true;
}

template<DeviceType::Type MType>
bool Device<MType>::
initializeFireTableUpdater_(DeviceDescription* description) {
  std::vector<Synapse*> synapse_vector;
  for (auto plugin : description->getSynapsePlugins()) {
    for (auto synapse : plugin->getSynapses()) {
      synapse_vector.push_back(synapse);
    }
    size_t padded_size = Bit::pad(synapse_vector.size());
    while (synapse_vector.size() != padded_size) {
      synapse_vector.push_back(nullptr);
    }
  }
  fire_table_updater_ = new FireTableUpdater<MType>();
  if (!fire_table_updater_->init(fire_table_,
                                 global_vector_injector_,
                                 synapse_vector)) {
    std::cerr << "Failed to initialize FireTableUpdater." << std::endl;
    return false;
  }
  return true;
}

template<DeviceType::Type MType>
bool Device<MType>::
initializeInputUpdater_(SpecificPublisher<StepSignal>* signal_publisher) {
  input_updater_ = new InputUpdater<MType>();
  if (!input_updater_->init(signal_publisher,
                            Constants::num_buffers,
                            device_synaptic_vector_size_)) {
    std::cerr << "Failed to initialize InputUpdater." << std::endl;
    return false;
  }
  return true;
}

} // namespace sim

} // namespace ncs
