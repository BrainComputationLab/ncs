#pragma once
namespace ncs {

namespace sim {

template<DeviceType::Type MType>
class InputBuffer : public DataBuffer {
public:
  InputBuffer(size_t device_neuron_vector_size);
  bool init();
  Bit::Word* getVoltageClampBits();
  float* getVoltageClampValues();
  float* getInputCurrent();
  bool clear();
  ~InputBuffer();
private:
  size_t device_neuron_vector_size_;
  Bit::Word* voltage_clamp_bits_;
  float* clamp_voltage_values_;
  float* input_current_;
};

} // namespace sim

} // namespace ncs

#include <ncs/sim/InputBuffer.hpp>
