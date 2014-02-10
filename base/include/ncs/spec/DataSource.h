#pragma once

#ifndef SWIG
#include <cstddef>

#include <ncs/sim/DataSink.h>
#include <ncs/sim/DataType.h>
#endif // SWIG

namespace ncs {

namespace spec {

class DataSource {
public:
#ifndef SWIG
  DataSource(ncs::sim::DataSink* data_sink);
  DataSource();
  bool isValid();
  size_t getTotalNumberOfElements() const;
  size_t getNumberOfPaddingElements() const;
  size_t getNumberOfRealElements() const;
  sim::DataType::Type getDataType() const;
  const void* pull();
  void release();
  void unsubscribe();
#endif // SWIG
  ~DataSource();
private:
  ncs::sim::DataSink* data_sink_;
  ncs::sim::DataSink::Subscription* subscription_;
  ncs::sim::DataSinkBuffer* pulled_buffer_;
};

} // namespace spec

} // namespace ncs
