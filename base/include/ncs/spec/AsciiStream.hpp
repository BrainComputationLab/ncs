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


  // Attempt to open a tcp socket. Once successful communication
  // takes place, we can try sending over the report data
  try {

      // uses localhost address and some arbritrary free port
      ncs::spec::ClientSocket client_socket ( "localhost", 30000 );
      std::string reply;

      try {
        client_socket << "Test message";
        client_socket >> reply;
      }
      catch ( ncs::spec::SocketException& ) {}

      std::cout << "We received this response:\n\"" << reply << "\"\n";;
      }
  catch ( ncs::spec::SocketException& e ) {
      std::cout << "Exception was caught:" << e.description() << "\n";
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
