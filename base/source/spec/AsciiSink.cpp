#include <ncs/sim/Bit.h>
#include <ncs/spec/AsciiSink.h>
#include <ncs/spec/AsciiStream.h>

namespace ncs {

namespace spec {

AsciiStreamSink::AsciiStreamSink(DataSource* data_source,
                                 std::ostream& stream)
  : stream_(nullptr) {
  if (nullptr == data_source) {
    return;
  }
  switch(data_source->getDataType()) {
    case sim::DataType::Float:
      stream_ = new AsciiStream<float>(stream, data_source);
      break;
    case sim::DataType::Integer:
      stream_ = new AsciiStream<int>(stream, data_source);
      break;
    case sim::DataType::Bit:
      stream_ = new AsciiStream<sim::Bit>(stream, data_source);
      break;
    default:
      std::cerr << "Unknown DataType." << std::endl;
      return;
      break;
  };
}

AsciiStreamSink::~AsciiStreamSink() {
  if (stream_) {
    delete stream_;
  }
}

AsciiFileSink::AsciiFileSink(DataSource* data_source,
                             const std::string& path) {
  file_.open(path);
  if (!file_) {
    std::cerr << "Failed to open file " << path << std::endl;
    return;
  }
  stream_ = new AsciiStreamSink(data_source, file_);
}

AsciiFileSink::~AsciiFileSink() {
  if (file_) {
    file_.close();
  }
  if (stream_) {
    delete stream_;
  }
}

} // namespace spec

} // namespace ncs
