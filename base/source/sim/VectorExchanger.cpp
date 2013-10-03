#include <ncs/sim/VectorExchanger.h>

namespace ncs {

namespace sim {

VectorExchanger::VectorExchanger() {
}

bool VectorExchanger::init(SpecificPublisher<StepSignal>* signal_publisher,
                           size_t global_neuron_vector_size,
                           size_t num_buffers) {
  global_neuron_vector_size_ = global_neuron_vector_size;
  num_buffers_ = num_buffers;
  for (size_t i = 0; i < num_buffers; ++i) {
    auto buffer = 
      new GlobalNeuronStateBuffer<DeviceType::CPU>(global_neuron_vector_size_);
    if (!buffer->init()) {
      std::cerr << "Failed to initialize GlobalNeuronStateBuffer<CPU>" <<
        std::endl;
      delete buffer;
      return false;
    }
    addBlank(buffer);
  }
  step_subscription_ = signal_publisher->subscribe();
  return nullptr != step_subscription_;
}

bool VectorExchanger::start() {
  auto thread_function = [this]() {
    while(true) {
      auto step_signal = step_subscription_->pull();
      if (nullptr == step_signal) {
        return;
      }
      auto blank = this->getBlank();
      this->publish(blank);
      step_signal->release();
    }
  };
  thread_ = std::thread(thread_function);
  return true;
}

} // namespace sim

} // namespace ncs
