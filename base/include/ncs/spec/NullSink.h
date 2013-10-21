#pragma once
#include <ncs/spec/DataSource.h>

namespace ncs {

namespace spec {

class NullSink {
public:
  NullSink(DataSource* data_source);
  ~NullSink();
private:
  DataSource* data_source_;
  std::thread thread_;
};

} // namespace spec

} // namespace ncs
