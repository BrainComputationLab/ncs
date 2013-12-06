#include <ncs/sim/DeviceBase.h>

namespace ncs {

namespace sim {

// thread local initialization
__thread DeviceBase* DeviceBase::thread_device_ = nullptr;

bool DeviceBase::setThreadDevice(DeviceBase* device) {
  thread_device_ = device;
  return true;
}

DeviceBase* DeviceBase::getThreadDevice() {
  return thread_device_;
}

DeviceBase::~DeviceBase() {
}

} // namespace sim

} // namespace ncs

