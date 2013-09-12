#pragma once
#include <ncs/spec/ModelParameters.h>
#include <ncs/sim/Synapse.h>

namespace ncs {

namespace sim {

template<DeviceType::Type MemoryType>
class SynapseSimulator {
public:
  virtual void* createInstantiator(spec::ModelParameters* parameters) = 0;
  virtual bool addSynapse(Synapse* synapse) = 0;
  virtual bool initialize() = 0;
  virtual bool destroyInstantiator(void* instantiator) = 0;
private:
};

} // namespace sim

} // namespace ncs
