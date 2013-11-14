#include <string>

#include <ncs/sim/MPI.h>

namespace ncs {

namespace sim {

Communicator* Communicator::global() {
  MPI_Group global_group;
  MPI_Comm new_comm;
  MPI_Comm_group(MPI_COMM_WORLD, &global_group);
  int result = MPI_Comm_create(MPI_COMM_WORLD, global_group, &new_comm);
  if (MPI::ok(result)) {
    return new Communicator(new_comm);
  } else {
    std::cerr << "Failed to create global MPI group: " << 
      MPI::errorString(result) << std::endl;
    return nullptr;
  }
}

template<> bool Communicator::send(const std::string& v, int rank) {
  unsigned int length = v.length();
  if (!send(length, rank)) {
    std::cerr << "Failed to send string length." << std::endl;
    return false;
  }
  if (0 == length) {
    return true;
  }
  if (!send(v.c_str(), length, rank)) {
    std::cerr << "Failed to send string contents." << std::endl;
    return false;
  }
  return true;
}

template<> bool Communicator::recv(std::string& v, int rank) {
  unsigned int length = 0;
  if (!recv(length, rank)) {
    std::cerr << "Failed to recv string length." << std::endl;
    return false;
  }
  if (0 == length) {
    v = "";
    return true;
  }
  char* buf = new char[length];
  if (!recv(buf, length, rank)) {
    std::cerr << "Failed to recv string contents." << std::endl;
    return false;
  }
  v = std::string(buf, length);
  delete [] buf;
  return true;
}

template<> bool Communicator::bcast(std::string& v, int origin_rank) {
  unsigned int length = 0;
  if (origin_rank == getRank()) {
    length = v.length();
  }
  if (!bcast(length, origin_rank)) {
    std::cerr << "Failed to bcast string length." << std::endl;
    return false;
  }
  if (0 == length) {
    v = "";
    return true;
  }
  if (origin_rank == getRank()) {
    if (!bcast(v.c_str(), length, origin_rank)) {
      std::cerr << "Failed to bcast string contents." << std::endl;
      return false;
    }
  } else {
    char* buf = new char[length];
    if (!bcast(buf, length, origin_rank)) {
      std::cerr << "Failed to bcast string contents (recv)." << std::endl;
      delete [] buf;
      return false;
    }
    v = std::string(buf, length);
    delete [] buf;
  }
  return true;
}


int Communicator::getRank() const {
  return rank_;
}

int Communicator::getNumProcesses() const {
  return num_processes_;
}

Communicator::Communicator(MPI_Comm comm)
  : mpi_communicator_(comm) {
  MPI_Comm_rank(mpi_communicator_, &rank_);
  MPI_Comm_size(mpi_communicator_, &num_processes_);
}

std::string MPI::errorString(int code) {
  char buf[MPI_MAX_ERROR_STRING];
  int length = 0;
  MPI_Error_string(code, buf, &length);
  return std::string(buf, length);
}

bool MPI::ok(int code) {
  return MPI_SUCCESS == code;
}

bool MPI::initialize(int argc, char** argv) {
  int provided_thread_level = 0;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided_thread_level);
  int num_processes = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
  if (provided_thread_level != MPI_THREAD_MULTIPLE && num_processes > 1) {
    std::cerr << "MPI_THREAD_MULTIPLE is not supported for this version " <<
      "of MPI. Cannot continue in a cluster environment." << std::endl;
    return false;
  }
  return true;
}

bool MPI::finalize() {
  MPI_Finalize();
  return true;
};

} // namespace sim

} // namespace ncs
