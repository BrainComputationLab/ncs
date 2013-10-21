#include <ncs/spec/NullSink.h>

namespace ncs {

namespace spec {

NullSink::NullSink(DataSource* data_source)
  : data_source_(data_source) {
  if (nullptr == data_source_) {
    return;
  }
  auto thread_function = [this]() {
    while (true) {
      const void* data = data_source_->pull();
      if (nullptr == data) {
        break;
      }
      data_source_->release();
    }
  };
  thread_ = std::thread(thread_function);
}

NullSink::~NullSink() {
  if (thread_.joinable()) {
    thread_.join();
  }
  if (data_source_) {
    delete data_source_;
  }
}

} // namespace spec

} // namespace ncs
