#pragma once
#include <ncs/sim/InputBuffer.h>
#include <ncs/sim/SimulationProperties.h>
#include <ncs/sim/StepSignal.h>

namespace ncs {

namespace sim {

template<DeviceType::Type MType>
class InputUpdater : public SpecificPublisher<InputBuffer<MType>> {
public:
  InputUpdater();
  bool init(SpecificPublisher<StepSignal>* signal_publisher,
            size_t num_buffers,
            size_t device_neuron_vector_size);
  bool step();
  ~InputUpdater();
private:
  typename SpecificPublisher<StepSignal>::Subscription* step_subscription_;
};

} // namespace sim

} // namespace ncs

#include <ncs/sim/InputUpdater.hpp>
