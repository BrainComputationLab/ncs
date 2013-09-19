#include <ncs/sim/SimulationController.h>

namespace ncs {

namespace sim {

SimulationController::SimulationController() {
  auto blank = new StepSignal();
  addBlank_(blank);
}

bool SimulationController::step() {
  auto signal = getBlank_();
  publish(signal);
}

SimulationController::~SimulationController() {
}

} // namespace sim

} // namespace ncs
