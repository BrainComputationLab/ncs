#pragma once
#include <ncs/sim/Bit.h>
#include <ncs/spec/DataSource.h>
#include <ncs/spec/ClientSocket.h>

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
	AsciiStream(std::ostream& stream, const std::string report_name,
		DataSource* data_source);
  virtual ~AsciiStream();
private:
  std::ostream& stream_;
  DataSource* data_source_;
  std::thread thread_;
  std::string report_name_;
};

template<>
AsciiStream<sim::Bit>::AsciiStream(std::ostream& stream,
                                   DataSource* data_source);

} // namespace spec

} // namespace ncs

#include <ncs/spec/AsciiStream.hpp>
