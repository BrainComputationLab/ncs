#include <ncs/sim/GlobalFireVectorBuffer.h>

namespace ncs {

namespace sim {

GlobalFireVectorBuffer::GlobalFireVectorBuffer()
  : fire_bits_(nullptr) {
}

void GlobalFireVectorBuffer::setFireBits(const Bit::Word* fire_bits) {
  fire_bits_ = fire_bits;
  setPin_("neuron_fire", fire_bits_, DeviceType::CPU);
}

const Bit::Word* GlobalFireVectorBuffer::getFireBits() const {
  return fire_bits_;
}

GlobalFireVectorBuffer::~GlobalFireVectorBuffer() {
}

} // namespace sim

} // namespace ncs
