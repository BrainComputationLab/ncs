#pragma once

#include <ncs/sim/SynapticCurrentBuffer.h>

namespace ncs {

namespace sim {

template<DeviceType::Type MType>
class SynapseSimulatorUpdater 
  : public SpecificPublisher<SynapticCurrentBuffer<MType>> {
public:
private:
};

} // namespace sim

} // namespace ncs
