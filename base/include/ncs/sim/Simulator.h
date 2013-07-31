#pragma once

#include <ncs/spec/ModelSpecification.h>

namespace ncs {

namespace sim {

class Simulator {
public:
  Simulator(spec::ModelSpecification* model_specification);
  bool initialize(int argc, char** argv);
private:
  bool initializeMPI_(int argc, char** argv);
  int mpi_rank_;
  int num_processes_;
  spec::ModelSpecification* model_specification_;
};

} // namespace sim

} // namespace ncs
