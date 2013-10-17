#pragma once
#include <ncs/sim/DataBuffer.h>

namespace ncs {

namespace sim {

class Signal : public DataBuffer {
public:
  Signal();
  bool getStatus() const;
  void setStatus(bool status);
  virtual ~Signal();
private:
  bool status_;
};

} // namespace sim

} // namespace ncs
