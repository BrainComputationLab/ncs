#pragma once

#include <ncs/sim/MPI.h>
#include <ncs/sim/ReportDataBuffer.h>
#include <ncs/sim/Signal.h>

#include <thread>
#include <vector>

namespace ncs {

namespace sim {

class ReportReceiver : public SpecificPublisher<Signal> {
public:
  ReportReceiver();
  bool init(size_t byte_offset,
            size_t num_bytes,
            Communicator* communicator,
            int source_rank,
            SpecificPublisher<ReportDataBuffer>* destination_publisher,
            size_t num_buffers);
  bool start();
  virtual ~ReportReceiver();
private:
  size_t byte_offset_;
  size_t num_bytes_;
  Communicator* communicator_;
  int source_rank_;
  SpecificPublisher<ReportDataBuffer>::Subscription* destination_subscription_;
  std::thread thread_;
};

} // namespace sim

} // namespace ncs
