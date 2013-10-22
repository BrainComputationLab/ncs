#include <ncs/sim/VectorExchanger.h>

namespace ncs {

namespace sim {

VectorExchanger::VectorExchanger() {
}

bool VectorExchanger::init(SpecificPublisher<StepSignal>* signal_publisher,
                           size_t global_neuron_vector_size,
                           size_t num_buffers) {
  global_neuron_vector_size_ = global_neuron_vector_size;
  num_buffers_ = num_buffers;
  for (size_t i = 0; i < num_buffers; ++i) {
    auto buffer = 
      new GlobalNeuronStateBuffer<DeviceType::CPU>(global_neuron_vector_size_);
    if (!buffer->init()) {
      std::cerr << "Failed to initialize GlobalNeuronStateBuffer<CPU>" <<
        std::endl;
      delete buffer;
      return false;
    }
    addBlank(buffer);
  }
  step_subscription_ = signal_publisher->subscribe();
  return nullptr != step_subscription_;
}

bool VectorExchanger::start() {
  auto thread_function = [this]() {
    while(true) {
      auto step_signal = step_subscription_->pull();
      if (nullptr == step_signal) {
        return;
      }
      auto blank = this->getBlank();
      this->publish(blank);
      step_signal->release();
    }
  };
  thread_ = std::thread(thread_function);
  return true;
}

VectorExchanger::~VectorExchanger() {
  if (thread_.joinable()) {
    thread_.join();
  }
  if (step_subscription_) {
    delete step_subscription_;
  }
}

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

GlobalVectorPublisher::GlobalVectorPublisher()
  : source_subscription_(nullptr) {
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
  return true;
}

bool GlobalVectorPublisher::start() {
  auto thread_function = [this]() {
    Mailbox mailbox;
    std::vector<Signal*> dependent_signals;
    dependent_signals.resize(dependent_subscriptions_.size());
    while(true) {
      VectorExchangeBuffer* source_buffer = nullptr;
      source_subscription_->pull(&source_buffer, &mailbox);
      for (size_t i = 0; i < dependent_subscriptions_.size(); ++i) {
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
