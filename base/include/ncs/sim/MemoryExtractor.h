#pragma once
#include <vector>

#include <ncs/sim/DataType.h>
#include <ncs/sim/DeviceType.h>

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

} // namespace sim

} // namespace ncs

#include <ncs/sim/MemoryExtractor.hpp>
