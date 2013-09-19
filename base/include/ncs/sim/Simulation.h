#pragma once
#include <ncs/spec/ModelSpecification.h>

namespace ncs {

namespace sim {

class Simulation {
public:
  Simulation(spec::ModelSpecification* model_specification);
  bool init(const std::vector<std::string>& args);
private:
  spec::ModelSpecification* model_specification_;
  class Simulator* simulator_;
};

} // namespace sim

} // namespace ncs
