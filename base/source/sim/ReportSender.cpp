#include <ncs/sim/ReportSender.h>

namespace ncs {

namespace sim {

ReportSender::ReportSender() {
  source_subscription_ = nullptr;
  communicator_ = nullptr;
}

bool ReportSender::
init(Communicator* communicator,
     int destination_rank,
     const std::vector<SpecificPublisher<Signal>*>& dependents,
     SpecificPublisher<ReportDataBuffer>* source_publisher) {
  communicator_ = communicator;
  destination_rank_ = destination_rank;
  source_subscription_ = source_publisher->subscribe();
  for (auto dependent : dependents) {
    dependent_subscriptions_.push_back(dependent->subscribe());
  }
  return true;
}

bool ReportSender::start() {
  auto thread_function = [this]() {
    Mailbox mailbox;
    std::vector<Signal*> dependent_signals(dependent_subscriptions_.size());
    while(true) {
      // Get a pull request
      if (!communicator_->recvState(destination_rank_)) {
        break;
      }
      for (size_t i = 0; i < dependent_signals.size(); ++i) {
        dependent_signals[i] = nullptr;
        dependent_subscriptions_[i]->pull(dependent_signals.data() + i, 
                                          &mailbox);
      }
      ReportDataBuffer* source_buffer = nullptr;
      source_subscription_->pull(&source_buffer, &mailbox);
      if (!mailbox.wait(&source_buffer, &dependent_signals)) {
        source_subscription_->cancel();
        if (source_buffer) {
          source_buffer->release();
        }
        for (size_t i = 0; i < dependent_signals.size(); ++i) {
          dependent_subscriptions_[i]->cancel();
          if (dependent_signals[i]) {
            dependent_signals[i]->release();
          }
        }
        communicator_->sendInvalid(destination_rank_);
        break;
      }
      bool status = true;
      for (auto signal : dependent_signals) {
        status &= signal->getStatus();
        signal->release();
      }
      if (!status) {
        communicator_->sendInvalid(destination_rank_);
        break;
      }
      communicator_->send(static_cast<const char*>(source_buffer->getData()),
                          source_buffer->getSize(),
                          destination_rank_);
      source_buffer->release();
    }
    delete source_subscription_;
    source_subscription_ = nullptr;
    for (auto sub : dependent_subscriptions_) {
      delete sub;
    }
    dependent_subscriptions_.clear();
  };
  thread_ = std::thread(thread_function);
  return true;
}

ReportSender::~ReportSender() {
  if (thread_.joinable()) {
    thread_.join();
  }
  for (auto sub : dependent_subscriptions_) {
    delete sub;
  }
  if (source_subscription_) {
    delete source_subscription_;
  }
}

} // namespace sim

} // namespace ncs
