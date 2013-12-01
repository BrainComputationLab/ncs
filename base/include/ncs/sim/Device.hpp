#include <ncs/sim/Bit.h>
#include <ncs/sim/Parallel.h>

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
           FactoryMap<InputSimulator>* input_plugins,
           VectorExchangeController* vector_exchange_controller,
           GlobalVectorPublisher* global_vector_publisher,
           size_t global_neuron_vector_size,
           size_t global_neuron_vector_offset,
           SpecificPublisher<StepSignal>* signal_publisher,
           const spec::SimulationParameters* simulation_parameters) {
  simulation_parameters_ = simulation_parameters;
  if (!allocateUpdaters_()) {
    std::cerr << "Failed to allocate updaters." << std::endl;
    return false;
  }

  if (!initializeNeurons_(description, neuron_plugins)) {
    std::cerr << "Failed to initialize neurons." << std::endl;
    return false;
  }

  if (!initializeNeuronUpdater_()) {
    std::cerr << "Failed to initialize NeuronSimulatorUpdater." << std::endl;
    return false;
  }

  if (!initializeVectorExtractor_(vector_exchange_controller,
                                   global_neuron_vector_offset)) {
    std::cerr << "Failed to initialize DeviceVectorExtractor." << std::endl;
    return false;
  }

  if (!this->initializeVectorInjector_(global_vector_publisher,
                                       global_neuron_vector_size)) {
    std::cerr << "Failed to initialize GlobalVectorInjector." << std::endl;
    return false;
  }

  if (!initializeSynapses_(description, synapse_plugins)) {
    std::cerr << "Failed to initialize synapses." << std::endl;
    return false;
  }

  if (!initializeSynapseUpdater_()) {
    std::cerr << "Failed to initialize SynapseSimulatorUpdater." << std::endl;
    return false;
  }

  if (!initializeFireTable_()) {
    std::cerr << "Failed to initialize fire table." << std::endl;
    return false;
  }

  if (!initializeFireTableUpdater_(description)) {
    std::cerr << "Failed to initialize FireTableUpdater." << std::endl;
    return false;
  }

  if (!initializeInputUpdater_(signal_publisher, input_plugins)) {
    std::cerr << "Failed to initialize InputUpdater." << std::endl;
    return false;
  }
  return true;
}

template<DeviceType::Type MType>
bool Device<MType>::initializeReporters(int machine_location,
                                        int device_location,
                                        ReportManagers* report_managers) {
  auto neuron_manager = report_managers->getNeuronManager();
  bool result = true;
  result &= neuron_manager->addDescription("neuron_voltage",
                                           DataDescription(DataSpace::Device,
                                                           DataType::Float));
  result &= neuron_manager->addSource("neuron_voltage",
                                      machine_location,
                                      device_location,
                                      -1,
                                      neuron_simulator_updater_);
  result &= neuron_manager->addDescription("input_current",
                                           DataDescription(DataSpace::Device,
                                                           DataType::Float));
  result &= neuron_manager->addSource("input_current",
                                      machine_location,
                                      device_location,
                                      -1,
                                      input_updater_);
  result &= neuron_manager->addDescription("clamp_voltage",
                                           DataDescription(DataSpace::Device,
                                                           DataType::Float));
  result &= neuron_manager->addSource("clamp_voltage",
                                      machine_location,
                                      device_location,
                                      -1,
                                      input_updater_);
  result &= neuron_manager->addDescription("clamp_voltage_bit",
                                           DataDescription(DataSpace::Device,
                                                           DataType::Bit));
  result &= neuron_manager->addSource("clamp_voltage_bit",
                                      machine_location,
                                      device_location,
                                      -1,
                                      input_updater_);
  result &= neuron_manager->addDescription("synaptic_current",
                                           DataDescription(DataSpace::Device,
                                                           DataType::Float));
  result &= neuron_manager->addSource("synaptic_current",
                                      machine_location,
                                      device_location,
                                      -1,
                                      synapse_simulator_updater_);
  return result;
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
  std::function<bool()> thread_init = [this]() {
    return this->threadInit();
  };
  std::function<bool()> thread_destroy = [this]() {
    return this->threadDestroy();
  };
  if (!input_updater_->start(thread_init, thread_destroy)) {
    std::cerr << "Failed to start InputUpdater." << std::endl;
    return false;
  }

  if (!neuron_simulator_updater_->start(thread_init,
                                        thread_destroy)) {
    std::cerr << "Failed to start NeuronSimulatorUpdater." << std::endl;
    return false;
  }

  if (!fire_vector_extractor_->start(thread_init, thread_destroy)) {
    std::cerr << "Failed to start DeviceVectorExtractor." << std::endl;
    return false;
  }

  if (!global_vector_injector_->start(thread_init, thread_destroy)) {
    std::cerr << "Failed to start GlobalVectorInjector." << std::endl;
    return false;
  }

  if (!fire_table_updater_->start(thread_init, thread_destroy)) {
    std::cerr << "Failed to start FireTableUpdater." << std::endl;
    return false;
  }

  if (!synapse_simulator_updater_->start(thread_init, thread_destroy)) {
    std::cerr << "Failed to start SynapseSimulatorUpdater." << std::endl;
    return false;
  }
  return true;
}

template<DeviceType::Type MType>
bool Device<MType>::addInput(const std::vector<Input*>& inputs,
                             void* instantiator,
                             const std::string& type,
                             float start_time,
                             float end_time) {
  return input_updater_->addInputs(inputs,
                                   instantiator,
                                   type,
                                   start_time,
                                   end_time);
}

template<DeviceType::Type MType>
SpecificPublisher<Signal>* Device<MType>::getVectorExtractor() {
  return fire_vector_extractor_;
}

template<DeviceType::Type MType>
Device<MType>::~Device() {
  ParallelDelete pd;
  pd.add(input_updater_, "InputUpdater");
  pd.add(neuron_simulator_updater_, "NeuronSimulatorUpdater");
  pd.add(fire_vector_extractor_, "DeviceVectorExtractor");
  pd.add(global_vector_injector_, "GlobalVectorInjector");
  pd.add(fire_table_updater_, "FireTableUpdater");
  pd.add(synapse_simulator_updater_, "SynapseSimulatorUpdater");
  pd.wait();
}

template<DeviceType::Type MType>
bool Device<MType>::allocateUpdaters_() {
  input_updater_ = new InputUpdater<MType>();
  input_updater_->setDevice(this);
  neuron_simulator_updater_ = new NeuronSimulatorUpdater<MType>();
  neuron_simulator_updater_->setDevice(this);
  fire_table_updater_ = new FireTableUpdater<MType>();
  fire_table_updater_->setDevice(this);
  synapse_simulator_updater_ = new SynapseSimulatorUpdater<MType>();
  synapse_simulator_updater_->setDevice(this);
  fire_vector_extractor_ = new DeviceVectorExtractor<MType>();
  fire_vector_extractor_->setDevice(this);
  global_vector_injector_ = new GlobalVectorInjector<MType>();
  global_vector_injector_->setDevice(this);
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
    neuron_device_id_offset += 
      Bit::pad(plugin_description->getNeurons().size());
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
  if (!simulator->initialize(simulation_parameters_)) {
    std::cerr << "Failed to initialize a neuron simulator." << std::endl;
    return false;
  }
  return true;
}

template<DeviceType::Type MType>
bool Device<MType>::initializeNeuronUpdater_() {
  return neuron_simulator_updater_->init(input_updater_,
                                         synapse_simulator_updater_,
                                         neuron_simulators_,
                                         neuron_device_id_offsets_,
                                         neuron_device_vector_size_,
                                         Constants::num_buffers);
}

template<DeviceType::Type MType>
bool Device<MType>::
initializeVectorExtractor_(VectorExchangeController* controller,
                           size_t global_neuron_vector_offset) {
  return fire_vector_extractor_->init(global_neuron_vector_offset,
                                      Constants::num_buffers,
                                      neuron_simulator_updater_,
                                      controller);
}

template<DeviceType::Type MType>
bool Device<MType>::
initializeVectorInjector_(GlobalVectorPublisher* publisher,
                          size_t global_vector_size) {
  return global_vector_injector_->init(publisher,
                                       global_vector_size,
                                       Constants::num_buffers);
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
initializeSynapseUpdater_() {
  auto updater = synapse_simulator_updater_;
  if (!updater->setFireVectorPublisher(fire_table_updater_)) {
    std::cerr << "Failed to set FireVectorPublisher." << std::endl;
    return false;
  }
  if (!updater->setNeuronStatePublisher(neuron_simulator_updater_)) {
    std::cerr << "Failed to set NeuronStatePublisher." << std::endl;
    return false;
  }
  return synapse_simulator_updater_->init(synapse_simulators_,
                                          synapse_device_id_offsets_,
                                          simulation_parameters_,
                                          neuron_device_vector_size_,
                                          Constants::num_buffers);
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
initializeInputUpdater_(SpecificPublisher<StepSignal>* signal_publisher,
                        FactoryMap<InputSimulator>* input_plugins) {
  if (!input_updater_->init(signal_publisher,
                            Constants::num_buffers,
                            neuron_device_vector_size_,
                            input_plugins,
                            simulation_parameters_)) {
    std::cerr << "Failed to initialize InputUpdater." << std::endl;
    return false;
  }
  return true;
}

} // namespace sim

} // namespace ncs
