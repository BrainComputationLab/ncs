#pragma once
#include <map>

#include <ncs/sim/FactoryMap.h>
#include <ncs/sim/InputBuffer.h>
#include <ncs/sim/InputSimulator.h>
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
            size_t device_neuron_vector_size,
            FactoryMap<InputSimulator>* input_plugins);
  bool step();
  bool addInputs(const std::vector<Input*>& inputs,
                 void* instantiator,
                 const std::string& type);
  ~InputUpdater();
private:
  typename SpecificPublisher<StepSignal>::Subscription* step_subscription_;
  std::vector<InputSimulator<MType>*> simulators_;
  std::map<std::string, unsigned int> simulator_type_indices_;
};

} // namespace sim

} // namespace ncs

#include <ncs/sim/InputUpdater.hpp>
