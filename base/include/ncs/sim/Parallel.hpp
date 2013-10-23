#include <iostream>

namespace ncs {

namespace sim {

template<typename T>
bool ParallelDelete::add(T* pointer, std::string name) {
  if (nullptr == pointer) {
    return false;
  }
  auto thread_function = [pointer, name]() {
    std::clog << "Deleting " << name << std::endl;
    delete pointer;
    std::clog << "Deleted " << name << std::endl;
  };
  threads_.push_back(std::thread(thread_function));
  return true;
}

template<typename T>
bool ParallelDelete::add(const std::vector<T*>& pointers,
                         std::string name) {
  for (size_t i = 0; i < pointers.size(); ++i) {
    T* pointer = pointers[i];
    auto thread_function = [pointer, i, name]() {
      std::clog << "Deleting " << name << "[" << i << "]" << std::endl;
      delete pointer;
      std::clog << "Deleted " << name << "[" << i << "]" << std::endl;
    };
    threads_.push_back(std::thread(thread_function));
  }
  return true;
}

} // namespace sim

} // namespace ncs
