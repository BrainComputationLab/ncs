#pragma once
#include <vector>

#include <ncs/sim/DataType.h>
#include <ncs/sim/DeviceType.h>
#include <ncs/sim/Storage.h>

namespace ncs {

namespace sim {

class MemoryExtractor {
public:
  virtual bool extract(const void* source,
                       void* destination) = 0;
  virtual ~MemoryExtractor();
};

template<typename T>
class CPUExtractor : public MemoryExtractor {
public:
  CPUExtractor(const std::vector<unsigned int>& indices);
  virtual bool extract(const void* source, void* destination);
  virtual ~CPUExtractor();
private:
  std::vector<unsigned int> indices_;
};

template<>
bool CPUExtractor<Bit>::extract(const void* source, void* destination);

#ifdef NCS_CUDA

template<typename T>
class CUDAExtractor : public MemoryExtractor {
public:
  CUDAExtractor(const std::vector<unsigned int>& indices);
  virtual bool extract(const void* source, void* destination);
  virtual ~CUDAExtractor();
private:
  unsigned int* indices_;
  size_t num_indices_;
  typename Storage<T>::type* device_buffer_;
  size_t buffer_size_;
};

#endif // NCS_CUDA

} // namespace sim

} // namespace ncs

#include <ncs/sim/MemoryExtractor.hpp>
