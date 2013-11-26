#include <ncs/sim/Parallel.h>
#include <ncs/sim/ReportSyncer.h>

namespace ncs {

namespace sim {

WorkerReportSyncer::WorkerReportSyncer() {
}

bool WorkerReportSyncer::
init(const std::vector<PublisherExtractor*>& extractors,
     Communicator* communicator,
     ReportController* report_controller,
     ReportSender* report_sender) {
  publisher_extractors_ = extractors;
  communicator_ = communicator;
  report_controller_ = report_controller;
  report_sender_ = report_sender;
  return true;
}

bool WorkerReportSyncer::run() {
  // Get the latest step that any of the extractors can pull
  unsigned int latest_step = 0;
  bool status = true;
  for (auto extractor : publisher_extractors_) {
    unsigned int step = 0;
    if (!extractor->getStep(step)) {
      std::cerr << "Failed to get the step of an extractor." << std::endl;
      status = false;
    }
    latest_step = std::max(latest_step, step);
  }
  if (!communicator_->syncState(status)) {
    std::cerr << "Some extractor failed to sync." << std::endl;
    return false;
  }
  // Tell the master that step
  communicator_->send(latest_step, 0);
  // Receive the cluster-wide step
  communicator_->bcast(latest_step, 0);
  // Sync every extractor to that step
  for (auto extractor : publisher_extractors_) {
    status &= extractor->syncStep(latest_step);
  }

  if (!communicator_->syncState(status)) {
    std::cerr << "Failed to sync reporters." << std::endl;
    return false;
  }

  // Start everything
  if (report_sender_) {
    status &= report_sender_->start();
  }
  for (auto extractor : publisher_extractors_) {
    status &= extractor->start();
  }
  if (report_controller_) {
    status &= report_controller_->start();
  }
  return status;
}

WorkerReportSyncer::~WorkerReportSyncer() {
  ParallelDelete pd;
  pd.add(report_controller_, "ReportController");
  pd.add(publisher_extractors_, "PublisherExtractor");
  pd.add(report_sender_, "ReportSender");
  pd.wait();
  delete communicator_;
}

MasterReportSyncer::MasterReportSyncer() {
}

bool MasterReportSyncer::
init(const std::vector<PublisherExtractor*>& extractors,
     const std::vector<ReportReceiver*>& receivers,
     Communicator* communicator,
     ReportController* report_controller) {
  publisher_extractors_ = extractors;
  report_receivers_ = receivers;
  communicator_ = communicator;
  report_controller_ = report_controller;
  return true;
}

bool MasterReportSyncer::run() {
  // Get the latest step from the extractors
  unsigned int latest_step = 0;
  bool status = true;
  for (auto extractor : publisher_extractors_) {
    unsigned int step = 0;
    if (!extractor->getStep(step)) {
      std::cerr << "Failed to get the step of an extractor." << std::endl;
      status = false;
    }
    latest_step = std::max(latest_step, step);
  }
  if (!communicator_->syncState(status)) {
    std::cerr << "Some extractor failed to sync." << std::endl;
    return false;
  }
  // Get the latest step from all the worker nodes
  for (size_t i = 1; i < communicator_->getNumProcesses(); ++i) {
    unsigned int step = 0;
    communicator_->recv(step, i);
    latest_step = std::max(latest_step, step);
  }

  // Broadcast the latest overall step
  communicator_->bcast(latest_step, 0);

  // Sync extractors to that step
  for (auto extractor : publisher_extractors_) {
    status &= extractor->syncStep(latest_step);
  }

  if (!communicator_->syncState(status)) {
    std::cerr << "Failed to sync reporters." << std::endl;
    return false;
  }

  status &= report_controller_->start();
  for (auto extractor : publisher_extractors_) {
    status &= extractor->start();
  }
  for (auto receiver : report_receivers_) {
    status &= receiver->start();
  }
  // Start everything
  return status;
}

MasterReportSyncer::~MasterReportSyncer() {
  ParallelDelete pd;
  pd.add(publisher_extractors_, "PublisherExtractor");
  pd.add(report_receivers_, "ReportReceiver");
  pd.add(report_controller_, "ReportController");
  pd.wait();
  delete communicator_;
}

} // namespace sim

} // namespace ncs
