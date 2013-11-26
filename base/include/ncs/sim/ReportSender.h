#pragma once
#include <ncs/sim/MPI.h>
#include <ncs/sim/ReportDataBuffer.h>
#include <ncs/sim/Signal.h>

#include <thread>
#include <vector>

namespace ncs {

namespace sim {

class ReportSender {
public:
  ReportSender();
  bool init(Communicator* communicator,
            int destination_rank,
            const std::vector<SpecificPublisher<Signal>*>& dependents,
            SpecificPublisher<ReportDataBuffer>* source_publisher);
  bool start();
  ~ReportSender();
private:
  Communicator* communicator_;
  int destination_rank_;
  std::vector<SpecificPublisher<Signal>::Subscription*> 
    dependent_subscriptions_;
  typename SpecificPublisher<ReportDataBuffer>::Subscription*
    source_subscription_;
  std::thread thread_;
};

} // namespace sim

} // namespace ncs
