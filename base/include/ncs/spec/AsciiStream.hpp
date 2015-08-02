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
    bool connected = client_socket.bindWithoutThrow( "127.0.1.1", 8005 ); // THIS IS THE PORT THE DAEMON IS LISTENING ON
    std::cout << "Connection status: " << connected << std::endl;

    // testing username send
    if (connected) {
      std::string sim_identifier("testuser@gmail.com..regular_spiking_izh");
      std::string lenStr = std::to_string(sim_identifier.length());
      if (lenStr.length() < 3)
        lenStr = '0' + lenStr;

      client_socket << lenStr + sim_identifier;
    }

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

        // send the message
        if (connected) {
          client_socket << buffer;
        }
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

    // protobuf object instance
    SimData simData;
    std::string buffer;

    // Attempt to open socket to stream data
    ncs::spec::ClientSocket client_socket;
    bool connected = client_socket.bindWithoutThrow( "127.0.1.1", 8005 );
    std::cout << "Connection to daemon: " << connected << std::endl;

    // send simulation identifier
    if (connected) {
      std::string lenStr = std::to_string(report_name_.length());
      if (lenStr.length() < 3)
        lenStr = '0' + lenStr;

      client_socket << lenStr + report_name_;
    }

    size_t num_elements = data_source_->getTotalNumberOfElements();

    while (true) {
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

        // append the size of the message to the message
        /*std::string report_name_size = std::to_string(report_name_.length());
        if (report_name_.length() < 10)
          report_name_size = '0' + report_name_size;
        buffer = report_name_size + report_name_ + buffer;
        buffer = std::to_string(buffer.length()) + buffer;*/

        // send the message
        if (connected) {
          client_socket << buffer;
        }
      }

      stream_ << d[0];
      for (size_t i = 1; i < num_elements; ++i) {

        simData.set_data(d[i]);
        if (!simData.SerializeToString(&buffer)) {
          std::cout << "failed to serialize data\n";
          break;
        }
        else {

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
