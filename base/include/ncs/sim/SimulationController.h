#pragma once
#include <ncs/sim/StepSignal.h>

namespace ncs {

namespace sim {

class SimulationController : public SpecificPublisher<StepSignal> {
public:
  SimulationController();
  bool step();
  bool idle();
  virtual ~SimulationController();
private:
  StepSignal* queued_blank_;
};

} // namespace sim

} // namespace ncs
