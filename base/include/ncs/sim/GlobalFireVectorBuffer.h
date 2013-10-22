#pragma once
#include <ncs/sim/Bit.h>
#include <ncs/sim/DataBuffer.h>

namespace ncs {

namespace sim {

class GlobalFireVectorBuffer : public DataBuffer {
public:
  GlobalFireVectorBuffer();
  void setFireBits(const Bit::Word* fire_bits);
  const Bit::Word* getFireBits() const;
  ~GlobalFireVectorBuffer();
private:
  const Bit::Word* fire_bits_;
};

} // namespace sim

} // namespace ncs
