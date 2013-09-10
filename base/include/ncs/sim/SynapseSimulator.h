#pragma once
#include <ncs/spec/ModelParameters.h>

namespace ncs {

namespace sim {

template<DeviceType::Type MemoryType>
class SynapseSimulator {
public:
  virtual void* createInstantiator(spec::ModelParameters* parameters) = 0;
  virtual bool addSynapse(void* instantiator, unsigned int seed) = 0;
  virtual bool initialize() = 0;
  virtual bool destroyInstantiator(void* instantiator) = 0;
private:
};

} // namespace sim

} // namespace ncs
