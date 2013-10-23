#include <ncs/sim/Parallel.h>

namespace ncs {

namespace sim {

ParallelDelete::ParallelDelete() {
}

bool ParallelDelete::wait() {
  for (auto& thread : threads_) {
    thread.join();
  }
  threads_.clear();
  return true;
}

ParallelDelete::~ParallelDelete() {
  wait();
}

} // namespace sim

} // namespace ncs
