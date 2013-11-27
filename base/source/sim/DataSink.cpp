#include <ncs/sim/DataSink.h>

namespace ncs {

namespace sim {

DataSinkBuffer::DataSinkBuffer()
  : data_(nullptr) {
}

void DataSinkBuffer::setData(const void* data) {
  data_ = data;
}

const void* DataSinkBuffer::getData() const {
  return data_;
}

DataSinkBuffer::~DataSinkBuffer() {
}

DataSink::DataSink(const DataDescription& data_description,
                   size_t num_padding_elements,
                   size_t num_real_elements,
                   size_t num_buffers)
  : data_description_(data_description),
    num_padding_elements_(num_padding_elements),
    num_real_elements_(num_real_elements),
    num_buffers_(num_buffers),
    source_subscription_(nullptr) {
  num_total_elements_ = num_padding_elements_ + num_real_elements_;
  report_syncer_ = nullptr;
}

bool DataSink::init(const std::vector<SpecificPublisher<Signal>*> dependents,
                    ReportController* report_controller,
                    MasterReportSyncer* report_syncer) {
  report_syncer_ = report_syncer;
  report_controller_ = report_controller;
  source_subscription_ = report_controller->subscribe();
  size_t data_size = DataType::num_bytes(num_total_elements_,
                                         data_description_.getDataType());
  if (data_size <= 0) {
    std::cerr << "No data is actually collected in this sink." << std::endl;
    return false;
  }
  if (num_buffers_ <= 0) {
    std::cerr << "num_buffers must be greater than 0 for DataSink." <<
      std::endl;
    return false;
  }
  for (size_t i = 0; i < num_buffers_; ++i) {
    DataSinkBuffer* buffer = new DataSinkBuffer();
    addBlank(buffer);
  }
  dependent_publishers_ = dependents;
  for (auto dependent : dependent_publishers_) {
    dependent_subscriptions_.push_back(dependent->subscribe());
  }
  return true;
}

bool DataSink::start() {
  auto syncer_function = [report_syncer_]() {
    report_syncer_->run();
    delete report_syncer_;
  };
  std::thread thread(syncer_function);
  thread.detach();
  report_syncer_ = nullptr;

  auto thread_function = [this]() {
    Mailbox mailbox;
    std::vector<Signal*> dependent_signals;
    dependent_signals.resize(dependent_subscriptions_.size());
    unsigned int simulation_step = 0;
    while(true) {
      ReportDataBuffer* data_buffer = nullptr;
      source_subscription_->pull(&data_buffer, &mailbox);
      for (size_t i= 0; i < dependent_subscriptions_.size(); ++i) {
        dependent_signals[i] = nullptr;
        dependent_subscriptions_[i]->pull(dependent_signals.data() + i,
                                          &mailbox);
      }
      if (!mailbox.wait(&data_buffer, &dependent_signals)) {
        source_subscription_->cancel();
        for (auto sub : dependent_subscriptions_) {
          sub->cancel();
        }
        if (data_buffer) {
          data_buffer->release();
        }
        for (auto signal : dependent_signals) {
          if (signal) {
            signal->release();
          }
        }
        break;
      }
      bool status = true;
      for (auto signal : dependent_signals) {
        status &= signal->getStatus();
        signal->release();
      }
      if (!status) {
        data_buffer->release();
        break;
      }
      
      auto blank = this->getBlank();
      blank->simulation_step = simulation_step;
      blank->setData(data_buffer->getData());
      auto prerelease_function = [data_buffer]() {
        data_buffer->release();
      };
      blank->setPrereleaseFunction(prerelease_function);
      unsigned int num_subscribers = this->publish(blank);
      if (0 == num_subscribers) {
        break;
      }
      ++simulation_step;
      // data_buffer will be automatically released upon publishing if zero
      // subscribers are listening, so don't release it here
    }
    clearSubscriptions();
  };
  thread_ = std::thread(thread_function);
  return true;
}

const DataDescription& DataSink::getDataDescription() const {
  return data_description_;
}

size_t DataSink::getTotalNumberOfElements() const {
  return num_total_elements_;
}

size_t DataSink::getNumberOfPaddingElements() const {
  return num_padding_elements_;
}

size_t DataSink::getNumberOfRealElements() const {
  return num_real_elements_;
}

DataSink::~DataSink() {
  if (thread_.joinable()) {
    thread_.join();
  }
  if (source_subscription_) {
    delete source_subscription_;
  }
  for (auto sub : dependent_subscriptions_) {
    delete sub;
  }
  if (report_syncer_) {
    delete report_syncer_;
  }
}

} // namespace sim

} // namespace ncs
