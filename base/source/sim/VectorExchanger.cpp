#include <ncs/sim/VectorExchanger.h>

namespace ncs {

namespace sim {

DeviceVectorExtractorBase::~DeviceVectorExtractorBase() {
}

MachineVectorExchanger::MachineVectorExchanger() {
}

bool MachineVectorExchanger::
init(size_t global_neuron_vector_size,
     size_t num_buffers,
     const std::vector<DeviceVectorExtractorBase*>& device_extractors,
     const std::vector<size_t>& neuron_global_id_offsets) {
  global_neuron_vector_size_ = global_neuron_vector_size;
  for (size_t i = 0; i < num_buffers; ++i) {
    auto buffer =
      new GlobalNeuronStateBuffer<DeviceType::CPU>(global_neuron_vector_size_);
    if (!buffer->init()) {
      std::cerr << "Failed to initialize GlobalNeuronStateBuffer" << std::endl;
      delete buffer;
      return false;
    }
    addBlank(buffer);
  }
  device_extractors_ = device_extractors;
  neuron_global_id_offsets_ = neuron_global_id_offsets;
  return true;
}

} // namespace sim

} // namespace ncs
