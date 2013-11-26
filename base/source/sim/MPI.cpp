#include <string>

#include <ncs/sim/MPI.h>

namespace ncs {

namespace sim {

int MPI::dependent_count_ = 0;
std::condition_variable MPI::state_changed_;
std::mutex MPI::mutex_;

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

bool Communicator::sendInvalid(int rank) {
  int result = MPI_Send(nullptr,
                        0,
                        MPI_CHAR,
                        rank,
                        Invalid,
                        mpi_communicator_);
  if (!MPI::ok(result)) {
    std::cerr << "Failed to send invalid tag: " << MPI::errorString(result) <<
      std::endl;
    return false;
  }
  return true;
}

bool Communicator::sendValid(int rank) {
  int result = MPI_Send(nullptr,
                        0,
                        MPI_CHAR,
                        rank,
                        Valid,
                        mpi_communicator_);
  if (!MPI::ok(result)) {
    std::cerr << "Failed to send valid tag: " << MPI::errorString(result) <<
      std::endl;
    return false;
  }
  return true;
}

bool Communicator::recvState(int rank) {
  MPI_Status status;
  int result = MPI_Recv(nullptr,
                        0,
                        MPI_CHAR,
                        rank,
                        MPI_ANY_TAG,
                        mpi_communicator_,
                        &status);
  if (!MPI::ok(result)) {
    std::cerr << "Failed to recv data: " << MPI::errorString(result) <<
      std::endl;
    return false;
  }
  return status.MPI_TAG == Valid;
}

bool Communicator::syncState(bool my_state) {
  bool result = my_state;
  if (getRank() == 0) {
    for (int i = 1; i < getNumProcesses(); ++i) {
      result &= recvState(i);
    }
  } else if (my_state) {
    sendValid(0);
  } else {
    sendInvalid(0);
  }
  bcast(result, 0);
  return result;
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
  MPI::addDependent();
}

Communicator::~Communicator() {
  MPI_Comm_free(&mpi_communicator_);
  MPI::removeDependent();
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
  std::unique_lock<std::mutex> lock(mutex_);
  while (dependent_count_ > 0) {
    state_changed_.wait(lock);
  }
  MPI_Finalize();
  return true;
};

bool MPI::addDependent() {
  std::unique_lock<std::mutex> lock(mutex_);
  dependent_count_++;
  state_changed_.notify_all();
  return true;
}

bool MPI::removeDependent() {
  std::unique_lock<std::mutex> lock(mutex_);
  dependent_count_--;
  state_changed_.notify_all();
  return true;
}

} // namespace sim

} // namespace ncs
