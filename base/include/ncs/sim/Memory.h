#pragma once

#include <ncs/sim/DeviceType.h>

namespace ncs {

namespace sim {

template<DeviceType::Type MType>
class Memory {
public:
  template<typename T> static T* malloc(size_t count);
  template<typename T> static bool malloc(T*& addr, size_t count);
  template<typename T> static bool free(T* addr);
  
  template<DeviceType::Type DestType>
  struct To {
    template<typename T>
    static bool copy(const T* src, T* dest, size_t count);
  };
private:
};

namespace mem {

  template<DeviceType::Type DestType, DeviceType::Type SourceType, typename T>
  bool copy(T* dst, const T* src, size_t count);

  template<DeviceType::Type DestType, DeviceType::Type SourceType, typename T>
  bool clone(T*& dst, const std::vector<T>& src);

} // namespace mem



} // namespace sim

} // namespace ncs

#include <ncs/sim/Memory.hpp>
