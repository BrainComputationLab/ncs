#pragma once
#include <ncs/sim/MPI.h>
#include <ncs/sim/PublisherExtractor.h>
#include <ncs/sim/ReportController.h>
#include <ncs/sim/ReportReceiver.h>
#include <ncs/sim/ReportSender.h>

namespace ncs {

namespace sim {

class WorkerReportSyncer {
public:
  WorkerReportSyncer();
  bool init(const std::vector<PublisherExtractor*>& extractors,
            Communicator* communicator,
            ReportController* report_controller,
            ReportSender* report_sender);
  bool run();
  ~WorkerReportSyncer();
private:
  std::vector<PublisherExtractor*> publisher_extractors_;
  ReportController* report_controller_;
  Communicator* communicator_;
  ReportSender* report_sender_;
};

class MasterReportSyncer {
public:
  MasterReportSyncer();
  bool init(const std::vector<PublisherExtractor*>& extractors,
            const std::vector<ReportReceiver*>& receivers,
            Communicator* communicator,
            ReportController* report_controller);
  bool run();
  ~MasterReportSyncer();
private:
  std::vector<PublisherExtractor*> publisher_extractors_;
  std::vector<ReportReceiver*> report_receivers_;
  ReportController* report_controller_;
  Communicator* communicator_;
};

} // namespace sim

} // namespace ncs
