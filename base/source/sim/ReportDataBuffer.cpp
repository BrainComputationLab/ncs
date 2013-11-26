#include <ncs/sim/ReportDataBuffer.h>

namespace ncs {

namespace sim {

ReportDataBuffer::ReportDataBuffer(size_t data_size)
  : data_size_(data_size),
    data_(nullptr) {
}

bool ReportDataBuffer::init() {
  if (data_size_ == 0) {
    return false;
  }
  data_ = new char[data_size_];
  return data_ != nullptr;
}

void* ReportDataBuffer::getData() const {
  return data_;
}

size_t ReportDataBuffer::getSize() const {
  return data_size_;
}

ReportDataBuffer::~ReportDataBuffer() {
  if (data_) {
    char* p = (char*)data_;
    delete [] p;
    data_ = nullptr;
  }
}

} // namespace sim

} // namespace ncs
