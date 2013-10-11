#pragma once
#include <ncs/spec/InputGroup.h>
#include <ncs/spec/ModelSpecification.h>
#include <ncs/spec/SimulationParameters.h>

namespace ncs {

namespace sim {

class Simulation {
public:
  Simulation(spec::ModelSpecification* model_specification);
  bool init(const std::vector<std::string>& args,
            spec::SimulationParameters* simulation_parameters);
  bool step();
  bool addInput(spec::InputGroup* input);
  bool shutdown();
  ~Simulation();
private:
  spec::ModelSpecification* model_specification_;
  class Simulator* simulator_;
};

} // namespace sim

} // namespace ncs
