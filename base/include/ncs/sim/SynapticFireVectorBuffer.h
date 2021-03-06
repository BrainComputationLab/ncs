#pragma once
#include <ncs/sim/DataBuffer.h>
#include <ncs/sim/DeviceType.h>

namespace ncs {

namespace sim {

template<DeviceType::Type MType>
class SynapticFireVectorBuffer : public DataBuffer {
public:
  SynapticFireVectorBuffer(size_t num_words);
  bool setData(Bit::Word* data_row);
  Bit::Word* getFireBits();
  bool init();
  size_t getNumberOfWords() const;
  ~SynapticFireVectorBuffer();
private:
  Bit::Word* data_row_;
  size_t num_words_;
};

} // namespace sim

} // namespace ncs

#include <ncs/sim/SynapticFireVectorBuffer.hpp>
