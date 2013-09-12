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
template<>
template<typename T>
bool Memory<DeviceType::CPU>::To<DeviceType::CPU>::copy(const T* src,
                                                        T* dest,
                                                        size_t count) {
  std::copy(src, src + count, dest);
  return true;
}

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
    return false;
  }
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
    return false;
  }
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
}

namespace mem {

template<DeviceType::Type DestType, DeviceType::Type SourceType, typename T>
bool copy(T* dst, T* src, size_t count) {
  return Memory<SourceType>::template
    To<DestType>::copy(src, dst, sizeof(T) * count);
}

} // namespace mem

} // namespace sim

} // namespace ncs
