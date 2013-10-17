#pragma once
#include <ncs/spec/DataSource.h>
#include <ncs/spec/InputGroup.h>
#include <ncs/spec/ModelSpecification.h>
#include <ncs/spec/Report.h>
#include <ncs/spec/SimulationParameters.h>

namespace ncs {

namespace sim {

class Simulation {
public:
  Simulation(spec::ModelSpecification* model_specification,
             spec::SimulationParameters* simulation_parameters);
  bool init(const std::vector<std::string>& args);
  bool step();
  bool addInput(spec::InputGroup* input);
  spec::DataSource* addReport(spec::Report* report);
  bool shutdown();
  ~Simulation();
private:
  spec::ModelSpecification* model_specification_;
  spec::SimulationParameters* simulation_parameters_;
  class Simulator* simulator_;
};

} // namespace sim

} // namespace ncs
