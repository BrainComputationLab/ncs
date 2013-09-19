#pragma once
#include <ncs/sim/InputBuffer.h>
#include <ncs/sim/SimulationProperties.h>

namespace ncs {

namespace sim {

template<DeviceType::Type MType>
class InputUpdater : public SpecificPublisher<InputBuffer<MType>> {
public:
  bool init(size_t num_buffers,
            size_t device_neuron_vector_size);
  bool step(SimulationProperties* properties);
private:
};

} // namespace sim

} // namespace ncs

#include <ncs/sim/InputUpdater.hpp>
