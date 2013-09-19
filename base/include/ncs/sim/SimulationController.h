#pragma once
#include <ncs/sim/StepSignal.h>

namespace ncs {

namespace sim {

class SimulationController : public SpecificPublisher<StepSignal> {
public:
  SimulationController();
  bool step();
  virtual ~SimulationController();
private:
};

} // namespace sim

} // namespace ncs
