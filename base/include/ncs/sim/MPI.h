#pragma once
#include <mpi.h>
#include <condition_variable>
#include <mutex>
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
  bool sendInvalid(int rank);
  bool sendValid(int rank);
  bool recvState(int rank);
  bool syncState(bool my_state);
  int getRank() const;
  int getNumProcesses() const;
  static Communicator* global();
  ~Communicator();
private:
  enum Status {
    Valid = 0,
    Invalid = 1
  };
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
  static bool finalize();
  static bool addDependent();
  static bool removeDependent();
private:
  static int dependent_count_;
  static std::condition_variable state_changed_;
  static std::mutex mutex_;
};

} // namespace sim

} // namespace ncs

#include <ncs/sim/MPI.hpp>

