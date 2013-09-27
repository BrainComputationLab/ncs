namespace ncs {

namespace sim {

template<DeviceType::Type MType>
InputUpdater<MType>::InputUpdater()
  : step_subscription_(nullptr) {
}

template<DeviceType::Type MType>
bool InputUpdater<MType>::init(SpecificPublisher<StepSignal>* signal_publisher,
                               size_t num_buffers,
                               size_t device_neuron_vector_size,
                               FactoryMap<InputSimulator>* input_plugins) {
  for (size_t i = 0; i < num_buffers; ++i) {
    auto buffer = new InputBuffer<MType>(device_neuron_vector_size);
    if (!buffer->init()) {
      std::cerr << "Failed to initialize InputBuffer." << std::endl;
      delete buffer;
      return false;
    }
    addBlank_(buffer);
  }
  step_subscription_ = signal_publisher->subscribe();
  if (nullptr == step_subscription_) {
    std::cerr << "Failed to subscribe to StepSignal generator." << std::endl;
    return false;
  }
  std::vector<std::string> input_types = input_plugins->getTypes();
  for (auto type : input_types) {
    auto simulator_generator = input_plugins->getProducer<MType>(type);
    if (!simulator_generator) {
      std::cerr << "Failed to get simulator for input type " << type <<
        std::endl;
      return false;
    }
    InputSimulator<MType>* simulator = simulator_generator();
    if (!simulator->initialize()) {
      std::cerr << "Failed to initialize InputSimulator for type " << type <<
        std::endl;
      delete simulator;
      return false;
    }
    simulator_type_indices_[type] = simulators_.size();
    simulators_.push_back(simulator);
  }
  return true;
}

template<DeviceType::Type MType>
bool InputUpdater<MType>::step() {
  auto step_signal = step_subscription_->pull();
  if (nullptr == step_signal) {
    return false;
  }
  auto buffer = this->getBlank_();
  this->publish(buffer);
  step_signal->release();
  return true;
}

template<DeviceType::Type MType>
bool InputUpdater<MType>::addInputs(const std::vector<Input*>& inputs,
                                    void* instantiator,
                                    const std::string& type,
                                    float start_time,
                                    float end_time) {
  auto search_result = simulator_type_indices_.find(type);
  if (simulator_type_indices_.end() == search_result) {
    std::cerr << "Failed to find an InputSimulator for type " << type <<
      std::endl;
    return false;
  }
  auto simulator = simulators_[search_result->second];
  return simulator->addInputs(inputs,
                              instantiator,
                              start_time,
                              end_time);
}


template<DeviceType::Type MType>
InputUpdater<MType>::~InputUpdater() {
  if (step_subscription_) {
    delete step_subscription_;
  }
}

} // namespace sim

} // namespace ncs
