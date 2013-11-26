#pragma once

#include <thread>

#include <ncs/sim/DataBuffer.h>
#include <ncs/sim/DataDescription.h>
#include <ncs/sim/ReportController.h>
#include <ncs/sim/ReportDataBuffer.h>
#include <ncs/sim/ReportSyncer.h>
#include <ncs/sim/Signal.h>

namespace ncs {

namespace sim {

class DataSinkBuffer : public DataBuffer {
public:
  DataSinkBuffer();
  void setData(const void* data);
  const void* getData() const;
  virtual ~DataSinkBuffer();
private:
  const void* data_;
};

class DataSink : public SpecificPublisher<DataSinkBuffer> {
public:
  DataSink(const DataDescription& data_description,
           size_t num_padding_elements,
           size_t num_real_elements,
           size_t num_buffers);
  bool init(const std::vector<SpecificPublisher<Signal>*> dependents,
            ReportController* report_controller,
            MasterReportSyncer* report_syncer);
  bool start();
  const DataDescription& getDataDescription() const;
  size_t getTotalNumberOfElements() const;
  size_t getNumberOfPaddingElements() const;
  size_t getNumberOfRealElements() const;
  virtual ~DataSink();
private:
  DataDescription data_description_;
  size_t num_padding_elements_;
  size_t num_real_elements_;
  size_t num_total_elements_;
  size_t num_buffers_;
  std::vector<SpecificPublisher<Signal>*> dependent_publishers_;
  typedef typename SpecificPublisher<Signal>::Subscription SignalSubscription;
  std::vector<SignalSubscription*> dependent_subscriptions_;
  ReportController* report_controller_;
  typename ReportController::Subscription* 
    source_subscription_;
  MasterReportSyncer* report_syncer_;
  std::thread thread_;
};

} // namespace sim

} // namespace ncs
