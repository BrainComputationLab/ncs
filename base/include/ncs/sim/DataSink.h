#pragma once
#include <ncs/sim/DataBuffer.h>
#include <ncs/sim/DataDescription.h>

namespace ncs {

namespace sim {

class DataSinkBuffer : public DataBuffer {
public:
  DataSinkBuffer(size_t data_size);
  bool init();
  const void* getData();
  virtual ~DataSinkBuffer();
private:
  size_t data_size_;
  void* data_;
};

class DataSink : public SpecificPublisher<DataSinkBuffer> {
public:
  DataSink(const DataDescription* data_description,
           size_t num_padding_elements,
           size_t num_real_elements,
           size_t num_buffers);
  bool init();
  const DataDescription* getDataDescription() const;
  size_t getTotalNumberOfElements() const;
  size_t getNumberOfPaddingElements() const;
  size_t getNumberOfRealElements() const;
  virtual ~DataSink();
private:
  DataDescription* data_description_;
  size_t num_padding_elements_;
  size_t num_real_elements_;
  size_t num_total_elements_;
  size_t num_buffers_;
};

} // namespace sim

} // namespace ncs
