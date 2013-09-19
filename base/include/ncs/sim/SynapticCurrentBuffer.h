#pragma once

#include <ncs/sim/DataBuffer.h>

namespace ncs {

namespace sim {

template<DeviceType::Type MType>
class SynapticCurrentBuffer : public DataBuffer {
public:
  SynapticCurrentBuffer(size_t device_neuron_vector_size);
  bool init();
  ~SynapticCurrentBuffer();
private:
  size_t device_neuron_vector_size_;
  float* current_per_neuron_;
};

} // namespace sim

} // namespace ncs

#include <ncs/sim/SynapticCurrentBuffer.hpp>
