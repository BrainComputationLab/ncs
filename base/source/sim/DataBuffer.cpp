#include <ncs/sim/DataBuffer.h>
#include <ncs/sim/Device.h>

namespace ncs {

namespace sim {

Mailbox::Mailbox()
	: failed(false) {
}

Publisher::Publisher() {
  // TODO(rvhoang): Set device
}

Publisher::Subscription* Publisher::subscribe() {
  // Generate a subscription object
  Subscription* sub = new Subscription(this);

  // Make sure no one else is touching the subscription list
  std::unique_lock<std::mutex> lock(mutex_);

  // Add the subscription to the subscriber list
  subscriptions_.push_back(sub);

  // Return the newly generated object
  return sub;
}

void Publisher::unsubscribe(Publisher::Subscription* sub) {
  // Make sure on one else is touching the subscription list
  std::unique_lock<std::mutex> lock(mutex_);

  // Remove the subscription
  subscriptions_.remove(sub);
}

unsigned int Publisher::publish(DataBuffer* db) {
  // Make sure no one else is touching the subscriber list
  std::unique_lock<std::mutex> lock(mutex_);

  // This publication will be sent to all subscribers
  db->subscription_count += subscriptions_.size();

  // Send the buffer out to all subscribers
  for (auto sub : subscriptions_)
    sub->push(db);

  // Return the number of subscribers we published to
  return subscriptions_.size();
}

bool Publisher::clearSubscriptions() {
  // Make sure no one else is touching the subscriber list
  std::unique_lock<std::mutex> lock(mutex_);
  // Invalidate all subscribers
  for (auto sub : subscriptions_) {
    sub->invalidate();
  }
  subscriptions_.clear();
  return true;
}

Publisher::~Publisher() {
}

DataBuffer::Pin::Pin()
	: data_(nullptr),
	  memory_type_(DeviceType::CPU) {
}

DataBuffer::Pin::Pin(const void* data, DeviceType::Type memory_type)
	: data_(data),
	  memory_type_(memory_type) {
}

const DataBuffer::Pin& DataBuffer::getPin(const std::string& key) const {
	auto pt = pins_.find(key);
	if (pt == pins_.end()) {
		std::cerr << "Pin " << key << " does not exist" << std::endl;
		return null_pin_;
	}
	return pins_.find(key)->second;
}

void DataBuffer::release() {
	release_function();
}

bool DataBuffer::update() {
	return 1 == std::atomic_fetch_sub(&updates_needed, 1u);
}

} // namespace sim

} // namespace ncs
