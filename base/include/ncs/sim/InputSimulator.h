#pragma once

#include <ncs/sim/DeviceType.h>
#include <ncs/sim/Input.h>
#include <ncs/sim/InputUpdateParameters.h>
#include <ncs/spec/ModelParameters.h>
#include <ncs/spec/SimulationParameters.h>

namespace ncs {

namespace sim {

template<DeviceType::Type MemoryType>
class InputSimulator {
public:
  virtual bool addInputs(const std::vector<Input*>& inputs,
                         void* instantiator,
                         float start_time,
                         float end_time) = 0;
  virtual bool initialize(const spec::SimulationParameters* parameters) = 0;

  virtual bool update(InputUpdateParameters* parameters) = 0;
private:
};

} // namespace sim

} // namespace ncs

