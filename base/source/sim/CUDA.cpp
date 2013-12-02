#ifdef NCS_CUDA
#include <iostream>

#include <ncs/sim/CUDA.h>

namespace ncs {

namespace sim {

__thread cudaStream_t CUDA::stream_;

bool CUDA::setDevice(int device_number) {
  if (cudaSuccess != cudaSetDevice(device_number)) {
    std::cerr << "cudaSetDevice failed: " <<
      cudaGetErrorString(cudaGetLastError()) << std::endl;
    std::cerr << "Device Number: " << device_number << std::endl;
    return false;
  }
  return true;
}

int CUDA::getDevice() {
  int device_number = -1;
  if (cudaSuccess != cudaGetDevice(&device_number)) {
    std::cerr << "cudaSetDevice failed: " <<
      cudaGetErrorString(cudaGetLastError()) << std::endl;
    return -1;
  }
  return device_number;
}

bool CUDA::initStream() {
  if (cudaSuccess != cudaStreamCreate(&stream_)) {
    std::cerr << "cudaStreamCreate failed: " <<
      cudaGetErrorString(cudaGetLastError()) << std::endl;
    return false;
  }
  return true;
}

cudaStream_t& CUDA::getStream() {
  return stream_;
}

bool CUDA::synchronize() {
  if (cudaSuccess != cudaStreamSynchronize(getStream())) {
    std::cerr << "cudaStreamSynchronize failed:" <<
      cudaGetErrorString(cudaGetLastError()) << std::endl;
    return false;
  }
  return true;
}

bool CUDA::endStream() {
  if (cudaSuccess != cudaStreamDestroy(stream_)) {
    std::cerr << "cudaStreamDestroy failed:" <<
      cudaGetErrorString(cudaGetLastError()) << std::endl;
    return false;
  }
  return true;
}

unsigned int CUDA::getThreadsPerBlock(size_t num_elements) {
  return 512;
}

unsigned int CUDA::getNumberOfBlocks(size_t num_elements) {
  unsigned int threads_per_block = getThreadsPerBlock(num_elements);
  return (num_elements + threads_per_block - 1) / threads_per_block;
}

} // namespace sim

} // namespace ncs
#endif
