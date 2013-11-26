#include <ncs/sim/VectorExchanger.h>

namespace ncs {

namespace sim {

VectorExchangeBuffer::VectorExchangeBuffer(size_t global_vector_size)
  : global_vector_size_(global_vector_size),
    data_vector_(nullptr) {
}

bool VectorExchangeBuffer::init() {
  if (global_vector_size_ > 0) {
    data_vector_ = new Bit::Word[Bit::num_words(global_vector_size_)];
  }
  return true;
}

Bit::Word* VectorExchangeBuffer::getData() {
  return data_vector_;
}

VectorExchangeBuffer::~VectorExchangeBuffer() {
  if (data_vector_) {
    delete [] data_vector_;
  }
}

VectorExchangeController::VectorExchangeController() {
}

bool VectorExchangeController::init(size_t global_vector_size,
                                    size_t num_buffers) {
  for (size_t i = 0; i < num_buffers; ++i) {
    auto buffer = new VectorExchangeBuffer(global_vector_size);
    if (!buffer->init()) {
      std::cerr << "Failed to initialize VectorExchangeBuffer." << std::endl;
      delete buffer;
      return false;
    }
    addBlank(buffer);
  }
  return true;
}

bool VectorExchangeController::start() {
  if (thread_.joinable()) {
    std::cerr << "VectorExchangeController already started." << std::endl;
    return false;
  }
  auto thread_function = [this]() {
    while(true) {
      auto blank = this->getBlank();
      auto num_subscribers = this->publish(blank);
      if (0 == num_subscribers) {
        return;
      }
    }
  };
  thread_ = std::thread(thread_function);
  return true;
}

VectorExchangeController::~VectorExchangeController() {
  if (thread_.joinable()) {
    thread_.join();
  }
}

RemoteVectorExtractor::RemoteVectorExtractor() {
  destination_subscription_ = nullptr;
}

bool RemoteVectorExtractor::init(size_t global_vector_offset,
                                 size_t machine_vector_size,
                                 Communicator* communicator, 
                                 int source_rank,
                                 DestinationPublisher* destination_publisher,
                                 size_t num_buffers) {
  num_buffers_ = num_buffers;
  for (size_t i = 0; i < num_buffers_; ++i) {
    auto blank = new Signal();
    addBlank(blank);
  }
  global_vector_offset_ = global_vector_offset;
  machine_vector_size_ = machine_vector_size;
  communicator_ = communicator;
  source_rank_ = source_rank;
  destination_subscription_ = destination_publisher->subscribe();
  return true;
}

bool RemoteVectorExtractor::start() {
  auto thread_function = [this]() {
    size_t word_offset = Bit::num_words(global_vector_offset_);
    size_t num_words = Bit::num_words(machine_vector_size_);
    while(true) {
      auto destination_buffer = destination_subscription_->pull();
      if (!communicator_->recv(destination_buffer->getData() + word_offset,
                               num_words,
                               source_rank_)) {
        destination_buffer->release();
        return;
      }
      auto signal = getBlank();
      publish(signal);
      destination_buffer->release();
    }
  };
  thread_ = std::thread(thread_function);
  return true;
}

RemoteVectorExtractor::~RemoteVectorExtractor() {
  if (thread_.joinable()) {
    thread_.join();
  }
  if (destination_subscription_) {
    delete destination_subscription_;
  }
}

RemoteVectorPublisher::RemoteVectorPublisher() {
  source_subscription_ = nullptr;
}

bool RemoteVectorPublisher::
init(size_t global_vector_offset,
     size_t machine_vector_size,
     Communicator* communicator,
     int destination_rank,
     SourcePublisher* source_publisher,
     const std::vector<DependentPublisher*>& dependent_publishers) {
  global_vector_offset_ = global_vector_offset;
  machine_vector_size_ = machine_vector_size;
  communicator_ = communicator;
  destination_rank_ = destination_rank;
  source_subscription_ = source_publisher->subscribe();
  for (auto pub : dependent_publishers) {
    dependent_subscriptions_.push_back(pub->subscribe());
  }
  return true;
}

bool RemoteVectorPublisher::start() {
  auto thread_function = [this]() {
    size_t word_offset = Bit::num_words(global_vector_offset_);
    size_t num_words = Bit::num_words(machine_vector_size_);
    Mailbox mailbox;
    std::vector<Signal*> dependent_signals(dependent_subscriptions_.size());
    while(true) {
      VectorExchangeBuffer* source_buffer = nullptr;
      source_subscription_->pull(&source_buffer, &mailbox);
      for (size_t i = 0; i < dependent_subscriptions_.size(); ++i) {
        dependent_signals[i] = nullptr;
        dependent_subscriptions_[i]->pull(dependent_signals.data() + i,
                                          &mailbox);
      }
      if (!mailbox.wait(&source_buffer, &dependent_signals)) {
        source_subscription_->cancel();
        for (auto sub : dependent_subscriptions_) {
          sub->cancel();
        }
        if (source_buffer) {
          source_buffer->release();
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
        source_buffer->release();
        break;
      }
      communicator_->send(source_buffer->getData() + word_offset,
                          num_words,
                          destination_rank_);
      source_buffer->release();
    }
    communicator_->sendInvalid(destination_rank_);
  };
  thread_ = std::thread(thread_function);
  return true;
}

RemoteVectorPublisher::~RemoteVectorPublisher() {
  if (thread_.joinable()) {
    thread_.join();
  }
  if (source_subscription_) {
    delete source_subscription_;
  }
  for (auto sub : dependent_subscriptions_) {
    delete sub;
  }
}

GlobalVectorPublisher::GlobalVectorPublisher()
  : source_subscription_(nullptr),
    initialized_(false) {
};

bool GlobalVectorPublisher::
init(size_t global_vector_size,
     size_t num_buffers,
     const std::vector<DependentPublisher*>& dependent_publishers,
     SpecificPublisher<VectorExchangeBuffer>* source_publisher) {
  for (size_t i = 0; i < num_buffers; ++i) {
    auto blank = new GlobalFireVectorBuffer();
    addBlank(blank);
  }
  for (auto dependent : dependent_publishers) {
    dependent_subscriptions_.push_back(dependent->subscribe());
  }
  source_subscription_ = source_publisher->subscribe();
  initialized_ = true;
  return true;
}

bool GlobalVectorPublisher::start() {
  if (thread_.joinable()) {
    std::cerr << "GlobalVectorPublisher already started." << std::endl;
    return false;
  }
  if (!initialized_) {
    std::cerr << "GlobalVectorPublisher not initialized." << std::endl;
    return false;
  }
  auto thread_function = [this]() {
    Mailbox mailbox;
    std::vector<Signal*> dependent_signals;
    dependent_signals.resize(dependent_subscriptions_.size());
    while(true) {
      VectorExchangeBuffer* source_buffer = nullptr;
      source_subscription_->pull(&source_buffer, &mailbox);
      for (size_t i = 0; i < dependent_subscriptions_.size(); ++i) {
        dependent_signals[i] = nullptr;
        dependent_subscriptions_[i]->pull(dependent_signals.data() + i,
                                          &mailbox);
      }
      if (!mailbox.wait(&source_buffer, &dependent_signals)) {
        source_subscription_->cancel();
        for (auto sub : dependent_subscriptions_) {
          sub->cancel();
        }
        if (source_buffer) {
          source_buffer->release();
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
        source_buffer->release();
        break;
      }

      auto blank = this->getBlank();
      blank->setFireBits(source_buffer->getData());
      auto prerelease_function = [source_buffer]() {
        source_buffer->release();
      };
      blank->setPrereleaseFunction(prerelease_function);
      unsigned int num_subscribers = this->publish(blank);
      if (0 == num_subscribers) {
        break;
      }
    }
  };
  thread_ = std::thread(thread_function);
  return true;
}

GlobalVectorPublisher::~GlobalVectorPublisher() {
  if (thread_.joinable()) {
    thread_.join();
  }
  if (source_subscription_) {
    delete source_subscription_;
  }
  for (auto sub : dependent_subscriptions_) {
    delete sub;
  }
}

} // namespace sim

} // namespace ncs
