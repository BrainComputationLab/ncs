#include <iostream>
#include <string.h>

#ifdef NCS_CUDA
#include <cuda.h>
#include <cuda_runtime.h>

#include <ncs/sim/CUDA.h>
#endif // NCS_CUDA

namespace ncs {

namespace sim {

template<DeviceType::Type MType>
template<typename T>
bool Memory<MType>::malloc(T*& addr, size_t count) {
  addr = Memory<MType>::template malloc<T>(count);
  return addr != nullptr;
}

#ifdef NCS_CUDA
template<>
template<typename T>
T* Memory<DeviceType::CUDA>::malloc(size_t count) {
  char* result = nullptr;
  if (cudaMalloc((void**)&result, sizeof(T) * count) != cudaSuccess) {
    std::cerr << "cudaMalloc failed: " <<
      cudaGetErrorString(cudaGetLastError()) << std::endl;
    return nullptr;
  }
  return (T*)result;
}

template<>
template<typename T>
bool Memory<DeviceType::CUDA>::free(T* addr) {
  if (addr) {
    if (cudaFree(addr) != cudaSuccess) {
      std::cerr << "cudaFree failed: " <<
        cudaGetErrorString(cudaGetLastError()) << std::endl;
      return false;
    }
  }
  return true;
}

template<>
template<typename T>
bool Memory<DeviceType::CUDA>::zero(T* addr, size_t count) {
  cudaError_t result = cudaMemsetAsync(addr,
                                       0,
                                       sizeof(T) * count,
                                       CUDA::getStream());
  if (cudaSuccess != result) {
    std::cerr << "cudaMemsetAsync failed: " <<
      cudaGetErrorString(cudaGetLastError()) << std::endl;
    return false;
  }
  return true;
}
#endif

template<>
template<typename T>
T* Memory<DeviceType::CPU>::malloc(size_t count) {
  return new T[count];
}

template<>
template<typename T>
bool Memory<DeviceType::CPU>::free(T* addr) {
  if (addr) {
    delete [] addr;
  }
  return true;
}

template<>
template<typename T>
bool Memory<DeviceType::CPU>::zero(T* addr, size_t count) {
  memset(addr, 0, sizeof(T) * count);
  return true;
}

template<>
template<>
template<typename T>
bool Memory<DeviceType::CPU>::To<DeviceType::CPU>::copy(const T* src,
                                                        T* dest,
                                                        size_t count) {
  std::copy(src, src + count, dest);
  return true;
}

#ifdef NCS_CUDA
template<>
template<>
template<typename T>
bool Memory<DeviceType::CPU>::To<DeviceType::CUDA>::copy(const T* src,
                                                         T* dest,
                                                         size_t count) {
  auto result = cudaMemcpyAsync(dest,
                                src,
                                count * sizeof(T),
                                cudaMemcpyHostToDevice,
                                ncs::sim::CUDA::getStream());
  if (cudaSuccess != result) {
    std::cerr << "cudaMemcpyAsync(cudaMemcpyHostToDevice) failed: " <<
      cudaGetErrorString(cudaGetLastError()) << std::endl;
    std::cerr << "Dest: " << dest << std::endl;
    std::cerr << "Source: " << src << std::endl;
    std::cerr << "Size: " << count * sizeof(T) << std::endl;
    std::cerr << "Device: " << CUDA::getDevice() << std::endl;
    return false;
  }
  return true;
}

template<>
template<>
template<typename T>
bool Memory<DeviceType::CUDA>::To<DeviceType::CPU>::copy(const T* src,
                                                         T* dest,
                                                         size_t count) {
  auto result = cudaMemcpyAsync(dest,
                                src,
                                count * sizeof(T),
                                cudaMemcpyDeviceToHost,
                                ncs::sim::CUDA::getStream());
  if (cudaSuccess != result) {
    std::cerr << "cudaMemcpyAsync(cudaMemcpyDeviceToHost) failed: " <<
      cudaGetErrorString(cudaGetLastError()) << std::endl;
    std::cerr << "Dest: " << dest << std::endl;
    std::cerr << "Source: " << src << std::endl;
    std::cerr << "Size: " << count * sizeof(T) << std::endl;
    return false;
  }
  return true;
}

template<>
template<>
template<typename T>
bool Memory<DeviceType::CUDA>::To<DeviceType::CUDA>::copy(const T* src,
                                                          T* dest,
                                                          size_t count) {
  auto result = cudaMemcpyAsync(dest,
                                src,
                                count * sizeof(T),
                                cudaMemcpyDeviceToDevice,
                                ncs::sim::CUDA::getStream());
  if (cudaSuccess != result) {
    std::cerr << "cudaMemcpyAsync(cudaMemcpyDeviceToDevice) failed: " <<
      cudaGetErrorString(cudaGetLastError()) << std::endl;
    return false;
  }
  return true;
}
#endif // NCS_CUDA

namespace mem {

template<DeviceType::Type DestType, DeviceType::Type SourceType, typename T>
bool copy(T* dst, const T* src, size_t count) {
  return Memory<SourceType>::template
    To<DestType>::copy(src, dst, count);
}

template<DeviceType::Type DestType, typename T>
bool clone(T*& dst, const std::vector<T>& src) {
  if (!Memory<DestType>::malloc(dst, src.size())) {
    return false;
  }
  if (!copy<DestType, DeviceType::CPU>(dst, src.data(), src.size())) {
    Memory<DestType>::free(dst);
    return false;
  }
  return true;
}

} // namespace mem

} // namespace sim

} // namespace ncs
