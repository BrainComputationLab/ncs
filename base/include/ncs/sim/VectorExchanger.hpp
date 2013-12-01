#include <ncs/sim/Memory.h>

namespace ncs {

namespace sim {

template<DeviceType::Type MType>
DeviceVectorExtractor<MType>::DeviceVectorExtractor() 
  : source_subscription_(nullptr),
    destination_subscription_(nullptr) {
}

template<DeviceType::Type MType>
bool DeviceVectorExtractor<MType>::init(size_t global_vector_offset,
          size_t num_buffers,
          SourcePublisher* source_publisher,
          DestinationPublisher* destination_publisher) {
  global_word_offset_ = Bit::num_words(global_vector_offset);
  for (size_t i = 0; i < num_buffers; ++i) {
    auto blank = new Signal();
    addBlank(blank);
  }
  source_subscription_ = source_publisher->subscribe();
  destination_subscription_ = destination_publisher->subscribe();
  return true;
}

template<DeviceType::Type MType>
bool DeviceVectorExtractor<MType>::
start(std::function<bool()> thread_init,
      std::function<bool()> thread_destroy) {
  if (thread_.joinable()) {
    std::cerr << "DeviceVectorExtractor already started." << std::endl;
    return false;
  }
  auto thread_function = [this, thread_init, thread_destroy]() {
    thread_init();
    while(true) {
      Mailbox mailbox;
      SourceBuffer* source_buffer = nullptr;
      source_subscription_->pull(&source_buffer, &mailbox);
      VectorExchangeBuffer* destination_buffer = nullptr;
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
        break;
      }
      auto signal = this->getBlank();
      Bit::Word* dst = destination_buffer->getData() + global_word_offset_;
      Bit::Word* src = source_buffer->getFireBits();
      size_t size = Bit::num_words(source_buffer->getVectorSize());
      bool result = mem::copy<DeviceType::CPU, MType>(dst, src, size);
      signal->setStatus(result);
      source_buffer->release();
      destination_buffer->release();
      auto num_subscribers = this->publish(signal);
      if (!result) {
        break;
      }
      if (0 == num_subscribers) {
        break;
      }
    }
    thread_destroy();
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
init(SpecificPublisher<GlobalFireVectorBuffer>* source_publisher,
     size_t global_neuron_vector_size,
     size_t num_buffers) {
  global_word_size_ = Bit::num_words(global_neuron_vector_size);
  for (size_t i = 0; i < num_buffers; ++i) {
    auto blank = new GlobalNeuronStateBuffer<MType>(global_neuron_vector_size);
    if (!blank->init()) {
      std::cerr << "Failed to initialize GlobalNeuronStateBuffer<MType>" <<
        std::endl;
      delete blank;
      return false;
    }
    addBlank(blank);
  }
  source_subscription_ = source_publisher->subscribe();
  return true;
}

template<DeviceType::Type MType>
bool GlobalVectorInjector<MType>::start(std::function<bool()> thread_init,
                                        std::function<bool()> thread_destroy) {
  if (thread_.joinable()) {
    std::cerr << "GlobalVectorInjector<MType> already started." << std::endl;
    return false;
  }
  auto thread_function = [this, thread_init, thread_destroy]() {
    thread_init();
    unsigned int simulation_step = 0;
    while(true) {
      auto source_buffer = source_subscription_->pull();
      if (nullptr == source_buffer) {
        break;
      }
      auto blank = this->getBlank();
      blank->simulation_step = simulation_step;
      auto dest = blank->getFireBits();
      auto src = source_buffer->getFireBits();
      if (!mem::copy<MType, DeviceType::CPU>(dest, src, global_word_size_)) {
        std::cerr << "Failed to inject buffer." << std::endl;
      }
      source_buffer->release();
      auto num_subscribers = this->publish(blank);
      if (0 == num_subscribers) {
        break;
      }
      ++simulation_step;
    }
    thread_destroy();
  };
  thread_ = std::thread(thread_function);
  return true;
}

template<DeviceType::Type MType>
GlobalVectorInjector<MType>::~GlobalVectorInjector() {
  if (thread_.joinable()) {
    thread_.join();
  }
  if (source_subscription_) {
    delete source_subscription_;
  }
}

} // namespace sim

} // namespace ncs
