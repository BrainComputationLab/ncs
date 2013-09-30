#include <ncs/sim/SimulationController.h>

namespace ncs {

namespace sim {

SimulationController::SimulationController()
  : queued_blank_(false) {
  auto blank = new StepSignal();
  addBlank(blank);
}

bool SimulationController::step() {
  StepSignal* signal = nullptr;
  if (queued_blank_) {
    signal = queued_blank_;
    queued_blank_ = nullptr;
  } else {
    signal = getBlank();
  }
  publish(signal);
}

bool SimulationController::idle() {
  if (queued_blank_) {
    return true;
  }
  queued_blank_ = getBlank();
  return true;
}

SimulationController::~SimulationController() {
  if (queued_blank_) {
    publish(queued_blank_);
    queued_blank_ = nullptr;
  }
}

} // namespace sim

} // namespace ncs
