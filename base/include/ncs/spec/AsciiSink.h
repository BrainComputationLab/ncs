#pragma once
#ifndef SWIG
#include <fstream>
#endif // SWIG
#include <ncs/spec/DataSource.h>

namespace ncs {

namespace spec {

class AsciiStreamSink {
public:
  AsciiStreamSink(DataSource* data_source,
                  std::ostream& stream = std::cout);
  ~AsciiStreamSink();
private:
  class AsciiStreamBase* stream_;
};

class AsciiFileSink {
public:
  AsciiFileSink(DataSource* data_source,
                const std::string& path);
  ~AsciiFileSink();
private:
  AsciiStreamSink* stream_;
  std::ofstream file_;
};

} // namespace spec

} // namespace ncs
