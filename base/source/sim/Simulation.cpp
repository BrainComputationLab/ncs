#include <ncs/sim/Simulation.h>
#include <ncs/sim/Simulator.h>

namespace ncs {

namespace sim {

Simulation::Simulation(spec::ModelSpecification* model_specification)
  : model_specification_(model_specification) {
  simulator_ = new Simulator(model_specification_);
}

bool Simulation::init(const std::vector<std::string>& args) {
  int argc = args.size();
  char** argv = new char*[argc + 1];
  for (int i = 0; i < argc; ++i) {
    argv[i] = (char*)args[i].c_str();
  }
  if (!simulator_->initialize(argc, argv)) {
    std::cerr << "Failed to intialize simulator." << std::endl;
    return false;
  }
  return true;
}

} // namespace sim

} // namespace ncs
