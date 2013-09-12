#pragma once

#include <ncs/sim/Bit.h>
#include <ncs/sim/DataBuffer.h>

namespace ncs {

namespace sim {

template<DeviceType::Type MType>
class DeviceNeuronStateBuffer : public DataBuffer {
public:
  DeviceNeuronStateBuffer(size_t device_neuron_vector_size);
  float* getVoltages();
  Bit::Word* getFireBits();
  bool isValid() const;
  ~DeviceNeuronStateBuffer();
private:
  float* voltages_;
  Bit::Word* fire_bits_;
  size_t device_neuron_vector_size_;
};

} // namespace sim

} // namespace ncs

#include <ncs/sim/DeviceNeuronStateBuffer.hpp>
