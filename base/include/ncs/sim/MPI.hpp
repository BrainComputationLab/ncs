#include <iostream>

namespace ncs {

namespace sim {

template<typename T>
bool Communicator::send(const T& v, int rank) {
  return send(&v, 1, rank);
}

template<typename T>
bool Communicator::send(const T* v, int count, int rank) {
  int result = MPI_Send((void*)v,
                        sizeof(T) * count,
                        MPI_CHAR,
                        rank,
                        Valid,
                        mpi_communicator_);
  if (!MPI::ok(result)) {
    std::cerr << "Failed to send data: " << MPI::errorString(result) <<
      std::endl;
    return false;
  }
  return true;
}

template<> bool Communicator::send(const std::string& v, int rank);

template<typename T>
bool Communicator::recv(T& v, int rank) {
  return recv(&v, 1, rank);
}

template<typename T>
bool Communicator::recv(T* v, int count, int rank) {
  MPI_Status status;
  int result = MPI_Recv((void*)v,
                        sizeof(T) * count,
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

template<> bool Communicator::recv(std::string& v, int rank);

template<typename T>
bool Communicator::bcast(T& v, int origin_rank) {
  return bcast(&v, 1, origin_rank);
}

template<typename T>
bool Communicator::bcast(T* v, int count, int origin_rank) {
  int result = MPI_Bcast((void*)v,
                         sizeof(T) * count,
                         MPI_CHAR,
                         origin_rank,
                         mpi_communicator_);
  if (!MPI::ok(result)) {
    std::cerr << "Failed to bcast data: " << MPI::errorString(result) <<
      std::endl;
    return false;
  }
  return true;
}

template<> bool Communicator::bcast(std::string& v, int origin_rank);

} // namespace sim

} // namespace ncs
