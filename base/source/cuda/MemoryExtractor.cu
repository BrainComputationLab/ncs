#include <stdio.h>

#include <ncs/cuda/MemoryExtractor.h>
#include <ncs/cuda/CUDA.h>
#include <ncs/sim/CUDA.h>

namespace ncs {

namespace sim {

namespace cuda {

template<typename T>
__global__ void extractKernel(const typename Storage<T>::type* source,
                              typename Storage<T>::type* destination,
                              const unsigned int* indices,
                              unsigned int num_indices) {
  unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int stride = blockDim.x * gridDim.x;
  while (index < num_indices) {
    destination[index] = source[indices[index]];
    index += stride;
  }
}

template<>
__global__ void extractKernel<Bit>(const Bit::Word* source,
                                   Bit::Word* destination,
                                   const unsigned int* indices,
                                   unsigned int num_indices) {
  unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int stride = blockDim.x * gridDim.x;
  extern __shared__ unsigned int result_vector[];
  unsigned int& warp_result = result_vector[threadIdx.x];
  unsigned int* result_vector_base = result_vector + 32 * warp::index();
  unsigned int warp_thread = warp::thread();
  unsigned int limit = math::ceiling(num_indices, 32);
  while (index < limit) {
    warp_result = 0;
    if (index < num_indices) {
      unsigned int i = indices[index];
      unsigned int bit = bit::bit(i);
      unsigned int word = source[bit::word(i)];
      warp_result = bit::extract(word, bit);
      warp_result >>= warp_thread;
    }
    warp::reduceOr(result_vector_base, warp_thread);
    if (warp::leader()) {
      destination[bit::word(index)] = warp_result;
    }
    index += stride;
  }
}

template<typename T>
void extract(const typename Storage<T>::type* source,
             typename Storage<T>::type* destination,
             const unsigned int* indices,
             unsigned int num_indices) {
  unsigned int threads_per_block = CUDA::getThreadsPerBlock(num_indices);
  unsigned int num_blocks = CUDA::getNumberOfBlocks(num_indices);
  extractKernel<T><<<num_blocks,
                     threads_per_block,
                     0,
                     CUDA::getStream()>>>(source, 
                                          destination, 
                                          indices, 
                                          num_indices);
  CUDA::synchronize();
}

template<>
void extract<Bit>(const typename Storage<Bit>::type* source,
                  typename Storage<Bit>::type* destination,
                  const unsigned int* indices,
                  unsigned int num_indices) {
  unsigned int threads_per_block = CUDA::getThreadsPerBlock(num_indices);
  unsigned int num_blocks = CUDA::getNumberOfBlocks(num_indices);
  unsigned int shared_memory =
    sizeof(unsigned int) * threads_per_block;
  extractKernel<Bit><<<num_blocks,
                        threads_per_block,
                        shared_memory,
                        CUDA::getStream()>>>(source, 
                                             destination, 
                                             indices, 
                                             num_indices);
  CUDA::synchronize();
}

template void extract<int>(const typename Storage<int>::type* source,
                        typename Storage<int>::type* destination,
                        const unsigned int* indices,
                        unsigned int num_indices);
template void extract<float>(const typename Storage<float>::type* source,
                        typename Storage<float>::type* destination,
                        const unsigned int* indices,
                        unsigned int num_indices);

} // namespace cuda

} // namespace sim

} // namespace ncs
