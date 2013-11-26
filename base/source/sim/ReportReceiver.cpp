#include <ncs/sim/ReportReceiver.h>

namespace ncs {

namespace sim {

ReportReceiver::ReportReceiver() {
}

bool ReportReceiver::
init(size_t byte_offset,
     size_t num_bytes,
     Communicator* communicator,
     int source_rank,
     SpecificPublisher<ReportDataBuffer>* destination_publisher,
     size_t num_buffers) {
  for (size_t i = 0; i < num_buffers; ++i) {
    addBlank(new Signal());
  }
  byte_offset_ = byte_offset;
  num_bytes_ = num_bytes;
  communicator_ = communicator;
  source_rank_ = source_rank;
  destination_subscription_ = destination_publisher->subscribe();
  return true;
}

bool ReportReceiver::start() {
  auto thread_function = [this]() {
    while(true) {
      auto destination_buffer = destination_subscription_->pull();
      if (!destination_buffer) {
        communicator_->sendInvalid(source_rank_);
        break;
      }
      communicator_->sendValid(source_rank_);
      char* dest = 
        static_cast<char*>(destination_buffer->getData()) + byte_offset_;
      bool remote_status = communicator_->recv(dest, num_bytes_, source_rank_);
      destination_buffer->release();
      auto blank = getBlank();
      if (!remote_status) {
        blank->setStatus(false);
        this->publish(blank);
        break;
      }
      if (0 == this->publish(blank)) {
        communicator_->sendInvalid(source_rank_);
        break;
      }
    }
    delete destination_subscription_;
    destination_subscription_ = nullptr;
  };
  thread_ = std::thread(thread_function);
  return true;
}

ReportReceiver::~ReportReceiver() {
  if (thread_.joinable()) {
    thread_.join();
  }
  if (destination_subscription_) {
    delete destination_subscription_;
  }
}

} // namespace sim

} // namespace ncs
