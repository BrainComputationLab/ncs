namespace ncs {

namespace sim {

template<DeviceType::Type MemoryType>
DeviceType::Type Device<MemoryType>::getDeviceType() const {
  return MemoryType;
}

} // namespace sim

} // namespace ncs
