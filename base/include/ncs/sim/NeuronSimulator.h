#pragma once

#include <functional> 

#include <ncs/sim/DeviceType.h>
#include <ncs/sim/Neuron.h>
#include <ncs/spec/ModelParameters.h>

namespace ncs {

namespace sim {

template<DeviceType::Type MemoryType>
class NeuronSimulator {
public:
  virtual bool addNeuron(Neuron* neuron) = 0;
  virtual bool initialize() = 0;
  virtual bool initializeVoltages(float* plugin_voltages) = 0;
private:
};

} // namespace sim

} // namespace ncs
