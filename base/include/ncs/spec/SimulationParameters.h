#pragma once

namespace ncs {

namespace spec {

class SimulationParameters {
public:
  SimulationParameters();
  bool setTimeStep(float time_step);
  float getTimeStep() const;
  bool setNeuronSeed(int s);
  int getNeuronSeed() const;
  bool setSynapseSeed(int s);
  int getSynapseSeed() const;
  ~SimulationParameters();
private:
  float time_step_;
  int neuron_seed_;
  int synapse_seed_;
};

} // namespace spec

} // namespace ncs
