#include <ncs/sim/Simulation.h>
#include <ncs/sim/Simulator.h>

namespace ncs {

namespace sim {

Simulation::Simulation(spec::ModelSpecification* model_specification,
                       spec::SimulationParameters* simulation_parameters)
  : model_specification_(model_specification),
    simulation_parameters_(simulation_parameters) {
  simulator_ = new Simulator(model_specification_, simulation_parameters_);
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

bool Simulation::step() {
  return simulator_->step();
}

bool Simulation::wait() {
  return simulator_->wait();
}

bool Simulation::addInput(spec::InputGroup* input) {
  return simulator_->addInput(input);
}

spec::DataSource* Simulation::addReport(spec::Report* report) {
  auto sink = simulator_->addReport(report);
  if (nullptr == sink) {
    return nullptr;
  }
  return new spec::DataSource(sink);
}

bool Simulation::shutdown() {
  if (simulator_) {
    delete simulator_;
    return true;
  }
  return false;
}

bool Simulation::isMaster() const {
  return simulator_->isMaster();
}

Simulation::~Simulation() {
  shutdown();
}

} // namespace sim

} // namespace ncs
