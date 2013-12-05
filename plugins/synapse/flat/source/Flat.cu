#include <stdio.h>
#include <ncs/cuda/CUDA.h>
#include <ncs/sim/CUDA.h>

#include "Flat.cuh"

namespace cuda {

__global__ void updateFlatKernel(const ncs::sim::Bit::Word* synaptic_fire,
                                 const unsigned int* device_neuron_device_ids,
                                 const float* device_current,
                                 float* synaptic_current,
                                 unsigned int num_synapses) {
  for (size_t i = grid::thread(); i < num_synapses; i += grid::stride()) {
    unsigned int fire_word_index = bit::word(i);
    unsigned int mask = bit::mask(i);
    if (synaptic_fire[fire_word_index] & mask) {
      atomicAdd(synaptic_current + device_neuron_device_ids[i],
                device_current[i]);
    }
  }
}

bool updateFlat(const ncs::sim::Bit::Word* synaptic_fire,
                const unsigned int* device_neuron_device_ids,
                const float* device_current,
                float* synaptic_current,
                unsigned int num_synapses) {
  using ncs::sim::CUDA;
  updateFlatKernel<<<CUDA::getNumberOfBlocks(num_synapses),
                     CUDA::getThreadsPerBlock(num_synapses),
                     0,
                     CUDA::getStream()>>>(synaptic_fire,
                                          device_neuron_device_ids,
                                          device_current,
                                          synaptic_current,
                                          num_synapses);
  return CUDA::synchronize();
}

} // namespace cuda
