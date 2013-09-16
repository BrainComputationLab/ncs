#include <ncs/sim/VectorExchanger.h>

namespace ncs {

namespace sim {

DeviceVectorExtractorBase::~DeviceVectorExtractorBase() {
}

MachineVectorExchanger::
MachineVectorExchanger(size_t global_neuron_vector_size,
                       size_t num_buffers)
  : global_neuron_vector_size_(global_neuron_vector_size),
    num_buffers_(num_buffers) {
}

bool MachineVectorExchanger::
init(const std::vector<DeviceVectorExtractorBase*>& device_extractors,
     const std::vector<size_t>& neuron_device_id_offsets) {
  for (size_t i = 0; i < num_buffers_; ++i) {
    auto buffer =
      new GlobalNeuronStateBuffer<DeviceType::CPU>(global_neuron_vector_size_);
    if (!buffer->isValid()) {
      delete buffer;
      return false;
    }
    addBlank_(buffer);
  }
  device_extractors_ = device_extractors;
  neuron_device_id_offsets_ = neuron_device_id_offsets;
  return true;
}

} // namespace sim

} // namespace ncs
