namespace ncs {

namespace spec {

template<typename T>
AsciiStream<T>::AsciiStream(std::ostream& stream,
                         DataSource* data_source)
  : stream_(stream),
    data_source_(data_source) {
  if (nullptr == data_source_) {
    return;
  }
  auto thread_function = [this]() {
    size_t num_elements = data_source_->getTotalNumberOfElements();
    while(true) {
      const void* data = data_source_->pull();
      if (nullptr == data) {
        break;
      }
      const T* d = static_cast<const T*>(data);
      stream_ << d[0];
      for (size_t i = 1; i < num_elements; ++i) {
        stream_ << " " << d[i];
      }
      stream_ << std::endl;
      data_source_->release();
    }
    delete data_source_;
    data_source_ = nullptr;
  };
  thread_ = std::thread(thread_function);
}

template<typename T>
AsciiStream<T>::~AsciiStream() {
  if (data_source_) {
    data_source_->unsubscribe();
  }
  if (thread_.joinable()) {
    thread_.join();
  }
  if (data_source_) {
    delete data_source_;
  }
}

} // namespace spec

} // namespace ncs
