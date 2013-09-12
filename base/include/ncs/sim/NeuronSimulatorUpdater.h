#pragma once

#include <ncs/sim/DeviceNeuronStateBuffer.h>
#include <ncs/sim/DeviceType.h>

namespace ncs {

namespace sim {

template<DeviceType::Type MType>
class NeuronSimulatorUpdater 
  : public SpecificPublisher<DeviceNeuronStateBuffer<MType>> {
public:
private:
};

} // namespace sim

} // namespace ncs
