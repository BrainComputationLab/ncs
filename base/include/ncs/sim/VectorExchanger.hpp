#include <ncs/sim/Memory.h>

namespace ncs {

namespace sim {

template<DeviceType::Type MType>
DeviceVectorExtractor<MType>::DeviceVectorExtractor() {
}

template<DeviceType::Type MType>
bool DeviceVectorExtractor<MType>::init(StatePublisher* publisher) {
  // TODO(rvhoang): uncomment this when implementing this part
  //state_subscription_ = publisher->subscribe();
  //return nullptr != state_subscription_;
  return true;
}

template<DeviceType::Type MType>
bool DeviceVectorExtractor<MType>::pull(Bit::Word* dst) {
  auto buffer = state_subscription_->pull();
  if (nullptr == buffer) {
    return false;
  }
  size_t num_words = Bit::num_words(buffer->getVectorSize());
  auto fire_bits = buffer->getFireBits();
  bool result = mem::copy<DeviceType::CPU, MType>(dst, fire_bits, num_words);
  buffer->release();
  return result;
}

template<DeviceType::Type MType>
DeviceVectorExtractor<MType>::~DeviceVectorExtractor() {
  delete state_subscription_;
}

template<DeviceType::Type MType>
GlobalVectorInjector<MType>::
GlobalVectorInjector(size_t global_neuron_vector_size,
                     size_t num_buffers)
  : global_neuron_vector_size_(global_neuron_vector_size),
    num_buffers_(num_buffers),
    subscription_(nullptr) {
}

template<DeviceType::Type MType>
bool GlobalVectorInjector<MType>::init(CPUGlobalPublisher* publisher) {
  for (size_t i = 0; i < num_buffers_; ++i) {
    auto buffer =
      new GlobalNeuronStateBuffer<MType>(global_neuron_vector_size_);
    if (!buffer->init()) {
      std::cerr << "Failed to initialize GlobalNeuronStateBuffer." << std::endl;
      delete buffer;
      return false;
    }
    this->addBlank(buffer);
  }
  subscription_ = publisher->subscribe();
  return nullptr != subscription_;
}

template<DeviceType::Type MType>
GlobalVectorInjector<MType>::~GlobalVectorInjector() {
  if (subscription_) {
    delete subscription_;
  }
}

} // namespace sim

} // namespace ncs
