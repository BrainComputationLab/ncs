#include <mpi.h>

#include <ncs/sim/Simulator.h>

namespace ncs {

namespace sim {

Simulator::Simulator(spec::ModelSpecification* model_specification)
  : mpi_rank_(-1),
    num_processes_(0),
    model_specification_(model_specification) {
}

bool Simulator::initialize(int argc, char** argv) {
  if (!initializeMPI_(argc, argv)) {
    std::cerr << "Failed to initialize MPI." << std::endl;
    return false;
  }
}

bool Simulator::initializeMPI_(int argc, char** argv) {
  int provided_thread_level = 0;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided_thread_level);
  MPI_Comm_size(MPI_COMM_WORLD, &num_processes_);
  if (provided_thread_level != MPI_THREAD_MULTIPLE && num_processes_ > 1) {
    std::cerr << "MPI_THREAD_MULTIPLE is not supported for this version " <<
      "of MPI. Cannot continue in a cluster environment." << std::endl;
    return false;
  }
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank_);
  return true;
}

} // namespace sim

} // namespace ncs
