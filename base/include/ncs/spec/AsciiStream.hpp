#include "SimData.pb.h"

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

    // protobuf object instance
    SimData simData;
    std::string buffer;

    // Attempt to open socket to stream data
    ncs::spec::ClientSocket client_socket;
    bool connected = client_socket.bindWithoutThrow( "127.0.1.1", 8003 );
    std::cout << "Connection status: " << connected << std::endl;

    size_t num_elements = data_source_->getTotalNumberOfElements();
    while(true) {
      const void* data = data_source_->pull();
      if (nullptr == data) {
        break;
      }
      const T* d = static_cast<const T*>(data);

      // serialize the data so it can be sent through a socket
      simData.set_data(d[0]);
      if (!simData.SerializeToString(&buffer)) {
        std::cout << "failed to serialize data\n";

        // jump ship until we determine how to proceed
        break;
      }
      else {

        // append the size of the message to the message
        buffer = std::to_string(buffer.length()) + buffer;

        // send the message
        if (connected)
          client_socket << buffer;
      }

      stream_ << d[0];
      for (size_t i = 1; i < num_elements; ++i) {

        // this is called when the data has multiple columns?
        simData.set_data(d[i]);
        if (!simData.SerializeToString(&buffer)) {
          std::cout << "failed to serialize data\n";
          break;
        }
        else {

        // append the size of the message to the message
          buffer = std::to_string(buffer.length()) + buffer;

        // send the message
          if (connected)
            client_socket << buffer;
        }
        stream_ << " " << d[i];
      }
      stream_ << std::endl;
      data_source_->release();
    }
    if (connected)
      client_socket.close();
    delete data_source_;
    data_source_ = nullptr;
  };
  thread_ = std::thread(thread_function);
}

template<typename T>
AsciiStream<T>::AsciiStream(std::ostream& stream, const std::string report_name,
                         DataSource* data_source)
  : stream_(stream),
    report_name_(report_name),
    data_source_(data_source) {
  if (nullptr == data_source_) {
    return;
  }

  auto thread_function = [this]() {

  //try {
      // protobuf object instance
      SimData simData;
      std::string buffer;
      
      // this is the address and port number the python side is listening on
      ncs::spec::ClientSocket client_socket ( "127.0.1.1", 8001 );
/*
  catch ( ncs::spec::SocketException& e ) {
      std::cout << "Exception:" << e.description() << "\n";
      }*/

    size_t num_elements = data_source_->getTotalNumberOfElements();
    while(true) {
      const void* data = data_source_->pull();
      if (nullptr == data) {
        break;
      }
      const T* d = static_cast<const T*>(data);

      // serialize the data so it can be sent through a socket
      simData.set_data(d[0]);
      if (!simData.SerializeToString(&buffer)) {
        std::cout << "failed to serialize data\n";
        break;
      }
      else {

        // append the size of the message and the report name to the message
        std::string report_name_size = std::to_string(report_name_.length());
        if (report_name_.length() < 10)
          report_name_size = '0' + report_name_size;
        buffer = report_name_size + report_name_ + buffer;
        buffer = std::to_string(buffer.length()) + buffer;

        // send the message
        client_socket << buffer;
      }

      stream_ << d[0];
      for (size_t i = 1; i < num_elements; ++i) {

        // when is this called?
        simData.set_data(d[i]);
        simData.SerializeToString(&buffer);
        client_socket << buffer;
        stream_ << " " << d[i];
      }
      stream_ << std::endl;
      data_source_->release();
    }
    client_socket.close();
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
