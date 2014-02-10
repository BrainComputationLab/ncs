#include <ncs/sim/DataSink.h>
#include <ncs/spec/DataSource.h>

namespace ncs {

namespace spec {

DataSource::DataSource(ncs::sim::DataSink* data_sink) 
  : data_sink_(data_sink),
    subscription_(nullptr),
    pulled_buffer_(nullptr) {
  if (data_sink_) {
    subscription_ = data_sink_->subscribe();
  }
  if (!data_sink_->start()) {
    std::cerr << "Failed to start DataSink." << std::endl;
    if (subscription_) {
      delete subscription_;
      subscription_ = nullptr;
    }
  }
}

DataSource::DataSource()
  : data_sink_(nullptr),
    subscription_(nullptr),
    pulled_buffer_(nullptr) {
}

bool DataSource::isValid() {
  return subscription_ != nullptr;
}

size_t DataSource::getTotalNumberOfElements() const {
  return data_sink_->getTotalNumberOfElements();
}

size_t DataSource::getNumberOfPaddingElements() const {
  return data_sink_->getNumberOfPaddingElements();
}

size_t DataSource::getNumberOfRealElements() const {
  return data_sink_->getNumberOfRealElements();
}

sim::DataType::Type DataSource::getDataType() const {
  return data_sink_->getDataDescription().getDataType();
}

const void* DataSource::pull() {
  release();
  pulled_buffer_ = subscription_->pull();
  if (nullptr == pulled_buffer_) {
    return nullptr;
  }
  return pulled_buffer_->getData();
}

void DataSource::release() {
  if (pulled_buffer_) {
    pulled_buffer_->release();
    pulled_buffer_ = nullptr;
  }
}

void DataSource::unsubscribe() {
  if (subscription_) {
    subscription_->unsubscribe();
  }
}

DataSource::~DataSource() {
  if (pulled_buffer_) {
    pulled_buffer_->release();
  }
  if (subscription_) {
    delete subscription_;
  }
  if (data_sink_) {
    delete data_sink_;
  }
}

} // namespace spec

} // namespace ncs
