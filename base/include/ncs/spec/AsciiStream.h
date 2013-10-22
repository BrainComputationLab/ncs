#pragma once
#include <ncs/sim/Bit.h>
#include <ncs/spec/DataSource.h>

namespace ncs {

namespace spec {

class AsciiStreamBase {
public:
  AsciiStreamBase();
  virtual ~AsciiStreamBase() = 0;
private:
};

template<typename T> 
class AsciiStream : public AsciiStreamBase {
public:
  AsciiStream(std::ostream& stream,
              DataSource* data_source);
  virtual ~AsciiStream();
private:
  std::ostream& stream_;
  DataSource* data_source_;
  std::thread thread_;
};

template<>
AsciiStream<sim::Bit>::AsciiStream(std::ostream& stream,
                                   DataSource* data_source);

} // namespace spec

} // namespace ncs

#include <ncs/spec/AsciiStream.hpp>
