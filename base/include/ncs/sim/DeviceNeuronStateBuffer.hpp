#include <ncs/sim/Memory.h>

namespace ncs {

namespace sim {

template<DeviceType::Type MType>
DeviceNeuronStateBuffer<MType>::
DeviceNeuronStateBuffer(size_t device_neuron_vector_size)
  : device_neuron_vector_size_(device_neuron_vector_size) {
  Memory<MType>::malloc(voltages_, device_neuron_vector_size_);
  Memory<MType>::malloc(fire_bits_, Bit::num_words(device_neuron_vector_size_));
  setPin_("neuron_voltage", voltages_, MType);
}

template<DeviceType::Type MType>
float* DeviceNeuronStateBuffer<MType>::getVoltages() {
  return voltages_;
}

template<DeviceType::Type MType>
Bit::Word* DeviceNeuronStateBuffer<MType>::getFireBits() {
  return fire_bits_;
}

template<DeviceType::Type MType>
bool DeviceNeuronStateBuffer<MType>::isValid() const {
  return nullptr != voltages_;
}

template<DeviceType::Type MType>
size_t DeviceNeuronStateBuffer<MType>::getVectorSize() const {
  return device_neuron_vector_size_;
}

template<DeviceType::Type MType>
DeviceNeuronStateBuffer<MType>::~DeviceNeuronStateBuffer() {
  Memory<MType>::free(voltages_);
  Memory<MType>::free(fire_bits_);
}

} // namespace sim

} // namespace ncs
