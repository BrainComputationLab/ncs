#pragma once

namespace ncs {

namespace spec {

class SimulationParameters {
public:
  SimulationParameters();
  bool setTimeStep(float time_step);
  float getTimeStep() const;
  ~SimulationParameters();
private:
  float time_step_;
};

} // namespace spec

} // namespace ncs
