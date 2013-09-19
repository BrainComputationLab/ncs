namespace ncs {

namespace sim {

template<DeviceType::Type MType>
InputUpdater<MType>::InputUpdater()
  : step_subscription_(nullptr) {
}

template<DeviceType::Type MType>
bool InputUpdater<MType>::init(SpecificPublisher<StepSignal>* signal_publisher,
                               size_t num_buffers,
                               size_t device_neuron_vector_size) {
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
InputUpdater<MType>::~InputUpdater() {
  if (step_subscription_) {
    delete step_subscription_;
  }
}

} // namespace sim

} // namespace ncs
