#include <ncs/sim/Memory.h>

namespace ncs {

namespace sim {

template<DeviceType::Type MType>
DeviceVectorExtractor<MType>::DeviceVectorExtractor()
  : source_subscription_(nullptr),
    destination_subscription_(nullptr) {
}

template<DeviceType::Type MType>
bool DeviceVectorExtractor<MType>::
setSourcePublisher(SourcePublisher* publisher) {
  source_subscription_ = publisher->subscribe();
  return nullptr != source_subscription_;
}

template<DeviceType::Type MType>
bool DeviceVectorExtractor<MType>::
setDestinationPublisher(DestinationPublisher* publisher) {
  destination_subscription_ = publisher->subscribe();
  return nullptr != destination_subscription_;
}

template<DeviceType::Type MType>
bool DeviceVectorExtractor<MType>::init(size_t global_word_offset,
                                        size_t num_buffers) {
  if (nullptr == destination_subscription_) {
    std::cerr << "Destination was not set for DeviceVectorExtractor." <<
      std::endl;
    return false;
  }
  if (nullptr == source_subscription_) {
    std::cerr << "Source was not set for DeviceVectorExtractor." << std::endl;
    return false;
  }
  global_word_offset_ = global_word_offset;
  num_buffers_ = num_buffers;
  for (size_t i = 0; i < num_buffers_; ++i) {
    addBlank(new ExchangeStatus());
  }
  return true;
}

template<DeviceType::Type MType>
bool DeviceVectorExtractor<MType>::start() {
  auto thread_function = [this](){
    Mailbox mailbox;
    while(true) {
      SourceBuffer* source = nullptr;
      source_subscription_->pull(&source, &mailbox);
      DestinationBuffer* destination = nullptr;
      destination_subscription_->pull(&destination, &mailbox);
      if (!mailbox.wait(&source, &destination)) {
        source_subscription_->cancel();
        destination_subscription_->cancel();
        if (source) {
          source->release();
        }
        if (destination) {
          destination->release();
        }
        auto status = this->getBlank();
        status->valid = false;
        this->publish(status);
        return;
      }
      auto status = this->getBlank();
      Bit::Word* dst = 
        destination->getFireBits() + Bit::num_words(global_word_offset_);
      Bit::Word* src = source->getFireBits();
      size_t size = Bit::num_words(source->getVectorSize());
      bool result = mem::copy<DeviceType::CPU, MType>(dst, src, size);
      status->valid = result;
      this->publish(status);
      source->release();
      destination->release();
      if (!result) {
        return;
      }
    }
  };
  thread_ = std::thread(thread_function);
  return true;
}

template<DeviceType::Type MType>
DeviceVectorExtractor<MType>::~DeviceVectorExtractor() {
  if (thread_.joinable()) {
    thread_.join();
  }
  if (source_subscription_) {
    delete source_subscription_;
  }
  if (destination_subscription_) {
    delete destination_subscription_;
  }
}

template<DeviceType::Type MType>
GlobalVectorInjector<MType>::GlobalVectorInjector()
  : source_subscription_(nullptr) {
}

template<DeviceType::Type MType>
bool GlobalVectorInjector<MType>::
init(const ExchangePublisherList& dependent_publishers,
     CPUGlobalPublisher* buffer_publisher,
     size_t global_neuron_vector_size,
     size_t num_buffers) {
  global_neuron_vector_size_ = global_neuron_vector_size;
  num_buffers_ = num_buffers;
  for (size_t i = 0; i < num_buffers_; ++i) {
    auto buffer = 
      new GlobalNeuronStateBuffer<MType>(global_neuron_vector_size_);
    if (!buffer->init()) {
      std::cerr << "Failed to initialize GlobalNeuronStateBuffer<MType>" <<
        std::endl;
      delete buffer;
      return false;
    }
    addBlank(buffer);
  }
  for (auto dependent : dependent_publishers) {
    dependent_subscriptions_.push_back(dependent->subscribe());
  }
  source_subscription_ = buffer_publisher->subscribe();
  return true;
}

template<DeviceType::Type MType>
bool GlobalVectorInjector<MType>::start() {
  auto thread_function = [this]() {
    Mailbox mailbox;
    std::vector<ExchangeStatus*> exchange_results;
    exchange_results.resize(dependent_subscriptions_.size());
    while(true) {
      GlobalNeuronStateBuffer<DeviceType::CPU>* cpu_buffer = nullptr;
      source_subscription_->pull(&cpu_buffer, &mailbox);
      for (size_t i = 0; i < dependent_subscriptions_.size(); ++i) {
        dependent_subscriptions_[i]->pull(exchange_results.data() + i,
                                          &mailbox);
      }
      if (!mailbox.wait(&cpu_buffer, &dependent_subscriptions_)) {
        source_subscription_->cancel();
        for (auto sub : dependent_subscriptions_) {
          sub->cancel();
        }
        if (cpu_buffer) {
          cpu_buffer->release();
        }
        for (auto result : exchange_results) {
          if (result) {
            result->release();
          }
        }
        return;
      }
      auto blank = this->getBlank();
      auto dest = blank->getFireBits();
      auto src = cpu_buffer->getFireBits();
      auto size = Bit::num_words(global_neuron_vector_size_);
      mem::copy<MType, DeviceType::CPU>(dest, src, size);
      this->publish(blank);
      cpu_buffer->release();
      for (auto result : exchange_results) {
        result->release();
      }
    }
  };
  thread_ = std::thread(thread_function);
  return true;
}

template<DeviceType::Type MType>
GlobalVectorInjector<MType>::~GlobalVectorInjector() {
  if (source_subscription_) {
    delete source_subscription_;
  }
  for (auto sub : dependent_subscriptions_) {
    delete sub;
  }
}

} // namespace sim

} // namespace ncs
