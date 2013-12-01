#include <ncs/sim/Memory.h>

#ifdef NCS_CUDA
#include <ncs/cuda/MemoryExtractor.h>
#endif // NCS_CUDA

namespace ncs {

namespace sim {

template<typename T>
CPUExtractor<T>::CPUExtractor(const std::vector<unsigned int>& indices)
  : indices_(indices) {
}

template<typename T>
bool CPUExtractor<T>::extract(const void* source, void* destination) {
  const T* s = static_cast<const T*>(source);
  T* d = static_cast<T*>(destination);
  for (size_t i = 0; i < indices_.size(); ++i) {
    unsigned int index = indices_[i];
    d[i] = s[index];
  }
  return true;
}

template<typename T>
CPUExtractor<T>::~CPUExtractor() {
}

#ifdef NCS_CUDA

template<typename T>
CUDAExtractor<T>::CUDAExtractor(const std::vector<unsigned int>& indices) {
  mem::clone<DeviceType::CUDA>(indices_, indices);
  num_indices_ = indices.size();
  buffer_size_ = Storage<T>::num_elements(num_indices_);
  Memory<DeviceType::CUDA>::malloc(device_buffer_, buffer_size_); 
}

template<typename T>
bool CUDAExtractor<T>::extract(const void* source, void* destination) {
  cuda::extract<T>(static_cast<const T*>(source),
                   device_buffer_,
                   indices_,
                   num_indices_);
  mem::copy<DeviceType::CPU, DeviceType::CUDA>(static_cast<T*>(destination),
                                               device_buffer_,
                                               buffer_size_);
  return true;
}

template<typename T>
CUDAExtractor<T>::~CUDAExtractor() {
  Memory<DeviceType::CUDA>::free(indices_);
  Memory<DeviceType::CUDA>::free(device_buffer_);
}

#endif // NCS_CUDA

} // namespace sim

} // namespace ncs
