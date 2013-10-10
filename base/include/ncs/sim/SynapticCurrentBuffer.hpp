namespace ncs {

namespace sim {

template<DeviceType::Type MType>
SynapticCurrentBuffer<MType>::
SynapticCurrentBuffer(size_t device_neuron_vector_size)
  : device_neuron_vector_size_(device_neuron_vector_size),
    current_per_neuron_(nullptr) {
}

template<DeviceType::Type MType>
bool SynapticCurrentBuffer<MType>::init() {
  if (device_neuron_vector_size_ > 0) {
    return Memory<MType>::malloc(current_per_neuron_,
                                 device_neuron_vector_size_);
  }
  return true;
}

template<DeviceType::Type MType>
float* SynapticCurrentBuffer<MType>::getCurrents() {
  return current_per_neuron_;
}

template<DeviceType::Type MType>
SynapticCurrentBuffer<MType>::~SynapticCurrentBuffer() {
  if (current_per_neuron_) {
    Memory<MType>::free(current_per_neuron_);
  }
}


} // namespace sim

} // namespace ncs
