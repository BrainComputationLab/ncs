#include <ncs/spec/SimulationParameters.h>

namespace ncs {

namespace spec {

SimulationParameters::SimulationParameters()
  : time_step_(0.001f) {
}

bool SimulationParameters::setTimeStep(float time_step) {
  time_step_ = time_step;
  return true;
}

float SimulationParameters::getTimeStep() const {
  return time_step_;
}

SimulationParameters::~SimulationParameters() {
}

} // namespace spec

} // namespace ncs
