#include <ncs/sim/Memory.h>

namespace ncs {

namespace sim {

template<DeviceType::Type MType>
DeviceVectorExtractor<MType>::DeviceVectorExtractor() {
}

template<DeviceType::Type MType>
bool DeviceVectorExtractor<MType>::init(StatePublisher* publisher) {
  state_subscription_ = publisher->subscribe();
  return nullptr != state_subscription_;
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

} // namespace sim

} // namespace ncs
