#pragma once

#include <functional> 

#include <ncs/sim/DeviceType.h>
#include <ncs/spec/ModelParameters.h>

namespace ncs {

namespace sim {

template<DeviceType::Type MemoryType>
class NeuronSimulator {
public:
  virtual bool addNeuron(void* instantiator, unsigned int seed) = 0;
  virtual bool initialize() = 0;
private:
};

} // namespace sim

} // namespace ncs
