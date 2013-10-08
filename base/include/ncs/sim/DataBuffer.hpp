namespace ncs {

namespace sim {

template<typename... Pointers>
bool Mailbox::wait(Pointers... pointers) {
  std::unique_lock<std::mutex> lock(mutex);
  while (any_null_(pointers...) && !failed) {
    arrival.wait(lock);
  }
  return !failed;
}

template<typename T, typename... OtherArgs>
bool Mailbox::any_null_(T t, OtherArgs... o) {
  return any_null_(t) || any_null_(o...); 
}

template<typename T>
bool Mailbox::any_null_(T** t) {
  return *t == nullptr;
}

template<typename T>
bool Mailbox::any_null_(std::vector<T*>* pointers) {
  for (auto p : *pointers) {
    if (nullptr == p)
      return true;
  }
  return false;
}

template<class PublicationType, class PublisherType>
Subscription<PublicationType, PublisherType>::
Subscription(PublisherType* publisher)
  : publisher_(publisher),
  mailbox_(nullptr),
  mailslot_(nullptr) {
}

template<class PublicationType, class PublisherType>
PublicationType* Subscription<PublicationType, PublisherType>::pull() {
  // Lock around us so only we can change our internal state
  std::unique_lock<std::mutex> lock(mutex_);
  // wait until a pub is available
  while (true) {
    // Do we have something in our mailbox?
    if (!deliveries_.empty()) {
      // Then return the oldest delivery
      auto result = deliveries_.front();
      deliveries_.pop();
      return result;
    }
    // Otherwise, make sure the publisher is still alive
    if (nullptr == publisher_) {
      return nullptr;
    }
    // otherwise, wait for something to happen, either a publication
    // or invalidation
    state_changed_.wait(lock);
  }
}

template<class PublicationType, class PublisherType>
void Subscription<PublicationType, PublisherType>::pull(PublicationType** slot,
                                                        Mailbox* mailbox) {
  *slot = nullptr;
  std::unique_lock<std::mutex> lock(mutex_);
  if (!deliveries_.empty()) {
    *slot = deliveries_.front();
    deliveries_.pop();
    return;
  }
  if (nullptr == publisher_) {
    mailbox->failed = true;
    return;
  }
  mailbox_ = mailbox;
  mailslot_ = slot;
}

template<class PublicationType, class PublisherType>
void Subscription<PublicationType, PublisherType>::cancel() {
  std::unique_lock<std::mutex> lock(mutex_);
  mailbox_ = nullptr;
  mailslot_ = nullptr;
}

template<class PublicationType, class PublisherType>
void Subscription<PublicationType, PublisherType>::push(PublicationType* data) {
  // Make sure we're the only one modifying the queue
  std::unique_lock<std::mutex> lock(mutex_);
  if (mailbox_) {
    std::unique_lock<std::mutex> lock(mailbox_->mutex);
    *mailslot_ = data;
    mailbox_->arrival.notify_all();
    mailbox_ = nullptr;
    mailslot_ = nullptr;
    return;
  }
  // Modify the queue with the new data
  deliveries_.push(data);
  // Inform any threads waiting for a pub that a new one is ready
  state_changed_.notify_all();
}

template<class PublicationType, class PublisherType>
void Subscription<PublicationType, PublisherType>::invalidate() {
  // Make sure no one else is trying to access the publisher variable
  std::unique_lock<std::mutex> lock(mutex_);

  // Set the publisher to null
  publisher_ = nullptr;

  // If a mailbox exists, fail it
  if (mailbox_) {
    std::unique_lock<std::mutex> lock(mailbox_->mutex);
    mailbox_->failed = true;
    mailbox_->arrival.notify_all();
  }

  // Inform waiting threads of the invalidation
  state_changed_.notify_all();
}

template<class PublicationType, class PublisherType>
Subscription<PublicationType, PublisherType>::~Subscription() {
  // Make sure a publisher can't invalidate itself before we unsubscribe
  std::unique_lock<std::mutex> lock(mutex_);
  auto publisher = publisher_;
  publisher_ = nullptr;
  lock.unlock();

  // If the publisher still exists
  if (publisher) {
    // Make sure we don't get any more publications to an unallocated sub
    publisher->unsubscribe(this);
  }

  lock.lock();
  // Release all remaining deliveries
  while (!deliveries_.empty()) {
    deliveries_.front()->release();
    deliveries_.pop();
  }
}

template<typename T>
SpecificPublisher<T>::SpecificPublisher()
: num_blanks_(0) {
}

template<typename T>
typename SpecificPublisher<T>::Subscription* SpecificPublisher<T>::subscribe() {
  // Generate a subscription
  Subscription* sub = new Subscription(this);

  // Make sure no one else is messing with the subscriber list
  std::unique_lock<std::mutex> lock(mutex_);

  // Add the new subscription to the list
  subscriptions_.push_back(sub);

  // Return the new subscription
  return sub;
}

template<typename T>
void SpecificPublisher<T>::publish(T* data) {
  // Make sure no one else is messing with the subscriber list
  std::unique_lock<std::mutex> lock(mutex_);
  // Increment the sub count by the number of specific subscriptions
  data->subscription_count += subscriptions_.size();
  // Then publish to the generic subscribers first, which adds its own count
  // and pushes data out
  unsigned int subpubs = Publisher::publish(data);
  // Push data out to our subscribers
  for (auto sub : subscriptions_) {
    sub->push(data);
  }
  // If no one is listening, just release the buffer
  if (subpubs == 0 && subscriptions_.empty()) {
    std::function<void()> prerelease_function = data->getPrereleaseFunction();
    if (prerelease_function) {
      prerelease_function();
    }
    blanks_.push(data);
    blank_available_.notify_all();
  }
}

template<typename T>
void SpecificPublisher<T>::unsubscribe(Subscription* sub) {
  // Make sure we're the only ones messing with the subscriber list
  std::unique_lock<std::mutex> lock(mutex_);
  // Find the subscription and remove it
  subscriptions_.remove(sub);
}

template<typename T>
SpecificPublisher<T>::~SpecificPublisher() {
  // Clear the generics
  Publisher::clearSubscriptions();
  {
    // Make sure no one else is touching the subscriber list
    std::unique_lock<std::mutex> lock(mutex_);

    // Invalidate all subscribers
    for (auto sub : subscriptions_)
      sub->invalidate();

    subscriptions_.clear();
  }

  // Destroy all blanks
  for (unsigned int i = 0; i < num_blanks_; ++i) {
    delete getBlank();
  }
}

template<typename T>
void SpecificPublisher<T>::addBlank(T* blank) {
  // Make sure the blank knows how to add itself back to the pile
  blank->release_function = [this, blank]() {
    if (blank->subscription_count == 0) {
      std::unique_lock<std::mutex> lock(this->mutex_);
      this->blanks_.push(blank);
      this->blank_available_.notify_all();
      return;
    }
    // Decrement the sub count
    unsigned int remaining =
      std::atomic_fetch_sub(&(blank->subscription_count), 1u);

    // If this call is the last one
    if (remaining == 1) {
      // Do any prerelease stuff
      std::function<void()> prerelease_function = 
        blank->getPrereleaseFunction();
      if (prerelease_function) {
        prerelease_function();
      }
      // Add ourself back onto the blank stack of the publisher
      std::unique_lock<std::mutex> lock(this->mutex_);
      this->blanks_.push(blank);
      this->blank_available_.notify_all();
    }
  };

  // Blanks have no subscribers to start
  blank->subscription_count = 0;

  // Add the blank to the stack
  std::unique_lock<std::mutex> lock(this->mutex_);
  this->blanks_.push(blank);
  ++this->num_blanks_;
}

template<typename T>
T* SpecificPublisher<T>::getBlank() {
  T* result = nullptr;
  std::unique_lock<std::mutex> lock(mutex_);
  // We must have a blank to return
  while (nullptr == result) {
    // If no blanks are available, wait and try again
    if (blanks_.empty()) {
      blank_available_.wait(lock);
    } else {
      // otherwise, grab the first one and get out
      result = blanks_.top();
      blanks_.pop();
    }
  }
  return result;
}

template<typename T>
bool DataBuffer::setPin_(const std::string& key, 
                         const T* data, 
                         DeviceType::Type memory_type) {
  pins_[key] = Pin(data, memory_type);
  return true;
}

} // namespace sim

} // namespace ncs
