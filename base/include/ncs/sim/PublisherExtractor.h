#pragma once
#include <thread>

#include <ncs/sim/DataType.h>
#include <ncs/sim/ReportDataBuffer.h>
#include <ncs/sim/Signal.h>

namespace ncs {

namespace sim {

class PublisherExtractor : public SpecificPublisher<Signal> {
public:
  PublisherExtractor();
  bool init(size_t output_offset,
            DataType::Type datatype,
            const std::vector<unsigned int>& indices,
            const std::string pin_name,
            Publisher* source_publisher,
            SpecificPublisher<ReportDataBuffer>* destination_publisher,
            unsigned int start_step,
            unsigned int end_step);
  bool getStep(unsigned int& step);
  bool syncStep(unsigned int step);
  bool start();
  virtual ~PublisherExtractor();
private:
  size_t output_offset_;
  DataType::Type datatype_;
  std::vector<unsigned int> indices_;
  Publisher* source_publisher_;
  Publisher::Subscription* source_subscription_;
  DataBuffer* queued_buffer_;
  typename SpecificPublisher<ReportDataBuffer>::Subscription* 
    destination_subscription_;
  std::string pin_name_;
  std::thread thread_;

  unsigned int start_step_;
  unsigned int end_step_;
};

} // namespace sim

} // namespace ncs
