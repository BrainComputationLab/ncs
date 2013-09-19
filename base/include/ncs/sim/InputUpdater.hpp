namespace ncs {

namespace sim {

template<DeviceType::Type MType>
bool InputUpdater<MType>::init(size_t num_buffers,
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
  return true;
}

} // namespace sim

} // namespace ncs
