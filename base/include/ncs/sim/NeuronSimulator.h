#pragma once

#include <functional> 

#include <ncs/sim/DeviceType.h>
#include <ncs/sim/Neuron.h>
#include <ncs/sim/NeuronUpdateParameters.h>
#include <ncs/spec/ModelParameters.h>
#include <ncs/spec/SimulationParameters.h>

namespace ncs {

namespace sim {

template<DeviceType::Type MemoryType>
class NeuronSimulator {
public:
  virtual bool addNeuron(Neuron* neuron) = 0;
  virtual bool initialize(const spec::SimulationParameters* parameters) = 0;
  virtual bool initializeVoltages(float* plugin_voltages) = 0;
  virtual bool update(NeuronUpdateParameters* parameters) = 0;
private:
};

} // namespace sim

} // namespace ncs
