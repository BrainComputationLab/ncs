#include <ncs/sim/ReportController.h>

namespace ncs {

namespace sim {

ReportController::ReportController() {
}

bool ReportController::init(size_t buffer_size,
                            size_t num_buffers) {
  num_buffers_ = num_buffers;
  buffer_size_ = buffer_size;
  for (size_t i = 0; i < num_buffers_; ++i) {
    auto buffer = new ReportDataBuffer(buffer_size_);
    if (!buffer->init()) {
      std::cerr << "Failed to initialize ReportDataBuffer." << std::endl;
      delete buffer;
      return false;
    }
    addBlank(buffer);
  }
}

bool ReportController::start() {
  auto thread_function = [this]() {
    while(true) {
      auto blank = this->getBlank();
      auto num_subscribers = this->publish(blank);
      if (0 == num_subscribers) {
        return;
      }
    }
  };
  thread_ = std::thread(thread_function);
  return true;
}

ReportController::~ReportController() {
  if (thread_.joinable()) {
    thread_.join();
  }
}

} // namespace sim

} // namespace ncs
