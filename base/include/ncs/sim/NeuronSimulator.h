#pragma once

#include <functional> 

#include <ncs/sim/DeviceType.h>
#include <ncs/sim/FactoryMap.h>
#include <ncs/spec/ModelParameters.h>

namespace ncs {

namespace sim {

template<DeviceType::Type MemoryType>
class NeuronSimulator {
public:
  virtual void* createInstantiator(spec::ModelParameters* parameters) = 0;
  virtual bool addNeuron(void* instantiator, unsigned int seed) = 0;
  virtual bool initialize() = 0;
  virtual bool destroyInstantiator(void* instantiator) = 0;
private:
};

typedef FactoryMap<NeuronSimulator> NeuronSimulatorFactoryMap;

} // namespace sim

} // namespace ncs
