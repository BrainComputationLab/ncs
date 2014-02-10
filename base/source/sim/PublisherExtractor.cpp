#include <ncs/sim/Constants.h>
#include <ncs/sim/Device.h>
#include <ncs/sim/MemoryExtractor.h>
#include <ncs/sim/PublisherExtractor.h>

namespace ncs {

namespace sim {

PublisherExtractor::PublisherExtractor()
  : output_offset_(0),
    datatype_(DataType::Unknown),
    source_subscription_(nullptr),
    destination_subscription_(nullptr) {
  queued_buffer_ = nullptr;
}

bool PublisherExtractor::
init(size_t output_offset,
     DataType::Type datatype,
     const std::vector<unsigned int>& indices,
     const std::string pin_name,
     Publisher* source_publisher,
     SpecificPublisher<ReportDataBuffer>* destination_publisher,
     unsigned int start_step,
     unsigned int end_step) {
  output_offset_ = output_offset;
  datatype_ = datatype;
  indices_ = indices;
  pin_name_ = pin_name;
  start_step_ = start_step;
  end_step_ = end_step;
  source_publisher_ = source_publisher;
  source_subscription_ = source_publisher_->subscribe();
  destination_subscription_ = destination_publisher->subscribe();
  for (size_t i = 0; i < Constants::num_buffers; ++i) {
    addBlank(new Signal());
  }
  return true;
}

bool PublisherExtractor::getStep(unsigned int& step) {
  queued_buffer_ = source_subscription_->pull();
  if (nullptr == queued_buffer_) {
    return false;
  }
  step = queued_buffer_->simulation_step;
  return true;
}

bool PublisherExtractor::syncStep(unsigned int step) {
  unsigned int current_step = queued_buffer_->simulation_step;
  while (current_step < step) {
    queued_buffer_->release();
    queued_buffer_ = source_subscription_->pull();
    if (nullptr == queued_buffer_) {
      return false;
    }
    current_step = queued_buffer_->simulation_step;
  }
  return true;
}

bool PublisherExtractor::start() {
  auto thread_function = [this]() {
    MemoryExtractor* extractor = nullptr;
    DeviceBase* device = source_publisher_->getDevice();
    if (device) {
      device->threadInit();
    }
    while(true) {
      DataBuffer* source_buffer = nullptr;
      if (queued_buffer_) {
        source_buffer = queued_buffer_;
        queued_buffer_ = nullptr;
      } else {
        source_buffer = source_subscription_->pull();
      }
      if (nullptr == source_buffer) {
        break;
      }
      if (source_buffer->simulation_step < start_step_) {
        source_buffer->release();
      } else {
        queued_buffer_ = source_buffer;
        break;
      }
    }
    while(true) {
      Mailbox mailbox;
      DataBuffer* source_buffer = nullptr;
      if (queued_buffer_) {
        source_buffer = queued_buffer_;
        queued_buffer_ = nullptr;
      } else {
        source_subscription_->pull(&source_buffer, &mailbox);
      }
      ReportDataBuffer* destination_buffer = nullptr;
      destination_subscription_->pull(&destination_buffer, &mailbox);
      if (!mailbox.wait(&source_buffer, &destination_buffer)) {
        source_subscription_->cancel();
        destination_subscription_->cancel();
        if (source_buffer) {
          source_buffer->release();
        }
        if (destination_buffer) {
          destination_buffer->release();
        }
        delete source_subscription_;
        source_subscription_ = nullptr;
        delete destination_subscription_;
        destination_subscription_ = nullptr;
        break;
      }
      if (source_buffer->simulation_step > end_step_) {
        source_buffer->release();
        destination_buffer->release();
        break;
      }
      auto signal = this->getBlank();
      auto pin = source_buffer->getPin(pin_name_);
      if (nullptr == extractor) {
        switch(pin.getMemoryType()) {
          case DeviceType::CPU:
            switch(datatype_) {
              case DataType::Float:
                extractor = new CPUExtractor<float>(indices_);
                break;
              case DataType::Integer:
                extractor = new CPUExtractor<int>(indices_);
                break;
              case DataType::Bit:
                extractor = new CPUExtractor<Bit>(indices_);
                break;
            }
            break;
#ifdef NCS_CUDA
          case DeviceType::CUDA:
            switch(datatype_) {
              case DataType::Float:
                extractor = new CUDAExtractor<float>(indices_);
                break;
              case DataType::Integer:
                extractor = new CUDAExtractor<int>(indices_);
                break;
              case DataType::Bit:
                extractor = new CUDAExtractor<Bit>(indices_);
                break;
            }
            break;
#endif // NCS_CUDA
          default:
            break;
        }
      }
      char* dest = 
        static_cast<char*>(destination_buffer->getData()) + output_offset_;
      extractor->extract(pin.getData(), dest);
      source_buffer->release();
      destination_buffer->release();
      auto num_subscribers = this->publish(signal);
      if (0 == num_subscribers) {
        break;
      }
    }
    if (device) {
      device->threadDestroy();
    }
    delete source_subscription_;
    source_subscription_ = nullptr;
    delete destination_subscription_;
    destination_subscription_ = nullptr;
    this->clearSubscriptions();
    if (extractor) {
      delete extractor;
    }
  };
  thread_ = std::thread(thread_function);
  return true;
}

PublisherExtractor::~PublisherExtractor() {
  if (thread_.joinable()) {
    thread_.join();
  }
  if (queued_buffer_) {
    queued_buffer_->release();
  }
  if (source_subscription_) {
    delete source_subscription_;
  }
  if (destination_subscription_) {
    delete destination_subscription_;
  }
}

} // namespace sim

} // namespace ncs
