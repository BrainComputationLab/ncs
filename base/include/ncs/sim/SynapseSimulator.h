#pragma once
#include <ncs/spec/ModelParameters.h>
#include <ncs/sim/Synapse.h>
#include <ncs/sim/SynapseUpdateParameters.h>

namespace ncs {

namespace sim {

template<DeviceType::Type MemoryType>
class SynapseSimulator {
public:
  virtual bool addSynapse(Synapse* synapse) = 0;
  virtual bool initialize() = 0;
  virtual bool update(SynapseUpdateParameters* parameters) = 0;
private:
};

} // namespace sim

} // namespace ncs
