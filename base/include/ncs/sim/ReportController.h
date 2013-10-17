#pragma once
#include <thread>

#include <ncs/sim/ReportDataBuffer.h>

namespace ncs {

namespace sim {

class ReportController : public SpecificPublisher<ReportDataBuffer> {
public:
  ReportController();
  bool init(size_t buffer_size,
            size_t num_buffers);
  bool start();
  virtual ~ReportController();
private:
  size_t num_buffers_;
  size_t buffer_size_;
  std::thread thread_;
};

} // namespace sim

} // namespace ncs
