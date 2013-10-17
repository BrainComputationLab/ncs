#pragma once
#include <ncs/sim/DataBuffer.h>

namespace ncs {

namespace sim {

class ReportDataBuffer : public DataBuffer {
public:
  ReportDataBuffer(size_t data_size);
  bool init();
  void* getData() const;
  virtual ~ReportDataBuffer();
private:
  size_t data_size_;
  void* data_;
};

} // namespace sim

} // namespace ncs
