#pragma once
namespace ncs {

namespace sim {

template<DeviceType::Type MType>
class InputBuffer : public DataBuffer {
public:
  InputBuffer(size_t device_neuron_vector_size);
  bool init();
  ~InputBuffer();
private:
};

} // namespace sim

} // namespace ncs
