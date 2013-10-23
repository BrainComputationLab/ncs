#pragma once
#include <thread>
#include <vector>

namespace ncs {

namespace sim {

class ParallelDelete {
public:
  ParallelDelete();
  template<typename T> bool add(T* pointer, std::string name);
  template<typename T> bool add(const std::vector<T*>& pointers,
                                std::string name);
  bool wait();
  ~ParallelDelete();
private:
  std::vector<std::thread> threads_;
};

} // namespace sim

} // namespace ncs

#include <ncs/sim/Parallel.hpp>
