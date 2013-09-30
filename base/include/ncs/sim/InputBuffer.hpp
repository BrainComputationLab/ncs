namespace ncs {

namespace sim {

template<DeviceType::Type MType>
InputBuffer<MType>::InputBuffer(size_t device_neuron_vector_size)
  : device_neuron_vector_size_(device_neuron_vector_size),
    voltage_clamp_bits_(nullptr),
    clamp_voltage_values_(nullptr),
    input_current_(nullptr) {
}

template<DeviceType::Type MType>
bool InputBuffer<MType>::init() {
  bool result = true;
  result &= Memory<MType>::malloc(voltage_clamp_bits_,
                                  Bit::num_words(device_neuron_vector_size_));
  result &= Memory<MType>::malloc(clamp_voltage_values_,
                                  device_neuron_vector_size_);
  result &= Memory<MType>::malloc(input_current_, device_neuron_vector_size_);
  if (!result) {
    std::cerr << "Failed to allocate memory for InputBuffer." << std::endl;
    return false;
  }
  return true;
}

template<DeviceType::Type MType>
bool InputBuffer<MType>::clear() {
  bool result = true;
  result &= Memory<MType>::zero(voltage_clamp_bits_,
                                Bit::num_words(device_neuron_vector_size_));
  result &= Memory<MType>::zero(clamp_voltage_values_,
                                device_neuron_vector_size_);
  result &= Memory<MType>::zero(input_current_, device_neuron_vector_size_);
  if (!result) {
    std::cerr << "Failed to clear memory for InputBuffer." << std::endl;
    return false;
  }
  return true;
}

template<DeviceType::Type MType>
Bit::Word* InputBuffer<MType>::getVoltageClampBits() {
  return voltage_clamp_bits_;
}

template<DeviceType::Type MType>
float* InputBuffer<MType>::getVoltageClampValues() {
  return clamp_voltage_values_;
}

template<DeviceType::Type MType>
float* InputBuffer<MType>::getInputCurrent() {
  return input_current_;
}

template<DeviceType::Type MType>
InputBuffer<MType>::~InputBuffer() {
  if (voltage_clamp_bits_) {
    Memory<MType>::free(voltage_clamp_bits_);
  }
  if (clamp_voltage_values_) {
    Memory<MType>::free(clamp_voltage_values_);
  }
  if (input_current_) {
    Memory<MType>::free(input_current_);
  }
}

} // namespace sim

} // namespace ncs
