namespace ncs {

namespace sim {

template<DeviceType::Type MType>
GlobalNeuronStateBuffer<MType>::
GlobalNeuronStateBuffer(size_t global_neuron_vector_size)
  : global_neuron_vector_size_(global_neuron_vector_size),
    fire_bits_(nullptr) {
}

template<DeviceType::Type MType>
bool GlobalNeuronStateBuffer<MType>::init() {
  if (global_neuron_vector_size_ > 0) {
    return Memory<MType>::malloc(fire_bits_, getNumberOfWords());
  }
  return 0 == global_neuron_vector_size_;
}

template<DeviceType::Type MType>
Bit::Word* GlobalNeuronStateBuffer<MType>::getFireBits() {
  return fire_bits_;
}

template<DeviceType::Type MType>
size_t GlobalNeuronStateBuffer<MType>::getVectorSize() const {
  return global_neuron_vector_size_;
}

template<DeviceType::Type MType>
size_t GlobalNeuronStateBuffer<MType>::getNumberOfWords() const {
  return Bit::num_words(global_neuron_vector_size_);
}

template<DeviceType::Type MType>
bool GlobalNeuronStateBuffer<MType>::isValid() const {
  return nullptr != fire_bits_;
}

template<DeviceType::Type MType>
GlobalNeuronStateBuffer<MType>::~GlobalNeuronStateBuffer() {
  Memory<MType>::free(fire_bits_);
}

} // namespace sim

} // namespace ncs
