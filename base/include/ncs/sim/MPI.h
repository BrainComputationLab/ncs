#pragma once
#include <mpi.h>
#include <string>

namespace ncs {

namespace sim {

class Communicator {
public:
  template<typename T> bool send(const T& v, int rank);
  template<typename T> bool send(const T* v, int count, int rank);
  template<typename T> bool recv(T& v, int rank);
  template<typename T> bool recv(T* v, int count, int rank);
  template<typename T> bool bcast(T& v, int origin_rank);
  template<typename T> bool bcast(T* v, int count, int origin_rank);
  int getRank() const;
  int getNumProcesses() const;
  static Communicator* global();
  ~Communicator();
private:
  Communicator(MPI_Comm comm);
  MPI_Comm mpi_communicator_;
  int rank_;
  int num_processes_;
};

class MPI {
public:
  static std::string errorString(int code);
  static bool ok(int code);
  static bool initialize(int argc, char** argv);
private:
};

} // namespace sim

} // namespace ncs

#include <ncs/sim/MPI.hpp>

