#pragma once
#include <ncs/spec/ModelSpecification.h>

namespace ncs {

namespace sim {

class Simulation {
public:
  Simulation(spec::ModelSpecification* model_specification);
  bool run(const std::vector<std::string>& args);
private:
  spec::ModelSpecification* model_specification_;
};

} // namespace sim

} // namespace ncs
