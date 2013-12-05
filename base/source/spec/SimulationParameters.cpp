#include <ncs/spec/SimulationParameters.h>

namespace ncs {

namespace spec {

SimulationParameters::SimulationParameters()
  : time_step_(0.001f) {
  neuron_seed_ = 0;
  synapse_seed_ = 0;
}

bool SimulationParameters::setTimeStep(float time_step) {
  time_step_ = time_step;
  return true;
}

float SimulationParameters::getTimeStep() const {
  return time_step_;
}

bool SimulationParameters::setNeuronSeed(int s) {
  neuron_seed_ = s;
}

int SimulationParameters::getNeuronSeed() const {
  return neuron_seed_;
}

bool SimulationParameters::setSynapseSeed(int s) {
  synapse_seed_ = s;
}

int SimulationParameters::getSynapseSeed() const {
  return synapse_seed_;
}

SimulationParameters::~SimulationParameters() {
}

} // namespace spec

} // namespace ncs
