#include <stdio.h>

#include <ncs/cuda/CUDA.h>
#include <ncs/sim/CUDA.h>
#include "Izhikevich.cuh"

namespace cuda {

__device__ float dvdt(float v, float u, float current) {
  return 0.04f * v * v + 5.0f * v + 140.0f - u + current;
}

__device__ float dudt(float v, float u, float a, float b) {
  return a * (b * v - u);
}

__global__ void updateNeuronsKernel(const float* as,
                                    const float* bs,
                                    const float* cs,
                                    const float* ds,
                                    const float* thresholds,
                                    const float* synaptic_current,
                                    const float* input_current,
                                    const float* old_u,
                                    const float* old_v,
                                    float* new_u,
                                    float* new_v,
                                    ncs::sim::Bit::Word* fire_vector,
                                    float step_dt,
                                    unsigned int num_neurons) {
	extern __shared__ ncs::sim::Bit::Word shared_fire_vector[];
	unsigned int& warp_result = shared_fire_vector[threadIdx.x];
	unsigned int* result_vector_base = shared_fire_vector + warp::index() * 32;
	unsigned int warp_thread = warp::thread();
  unsigned int limit = math::ceiling(num_neurons, 32);
	unsigned int mask = bit::mask(warp_thread);
  for (size_t i = grid::thread(); i < limit; i += grid::stride()) {
    warp_result = 0;
    if (i < num_neurons) {
      float a = as[i];
      float b = bs[i];
      float c = cs[i];
      float d = ds[i];
      float u = old_u[i];
      float v = old_v[i];
      float threshold = thresholds[i];
      float current = input_current[i] + synaptic_current[i];
      if (v >= threshold) {
        v = c;
        u += d;
        warp_result = mask;
      }
      float step_v = v + step_dt * dvdt(v, u, current);
      float step_u = u + step_dt * dudt(v, u, a, b);
      v = step_v;
      u = step_u;
      step_v = v + step_dt * dvdt(v, u, current);
      step_u = u + step_dt * dudt(v, u, a, b);
      v = step_v;
      u = step_u;
      if (v >= threshold) {
        v = threshold;
      }
      new_v[i] = v;
      new_u[i] = u;
    }
    warp::reduceOr(result_vector_base, warp_thread);
    if (warp::leader()) {
      fire_vector[bit::word(i)] = warp_result;
    }
  }
}

bool updateNeurons(const float* as,
                   const float* bs,
                   const float* cs,
                   const float* ds,
                   const float* thresholds,
                   const float* synaptic_current,
                   const float* input_current,
                   const float* old_u,
                   const float* old_v,
                   float* new_u,
                   float* new_v,
                   ncs::sim::Bit::Word* fire_vector,
                   float step_dt,
                   unsigned int num_neurons) {
  using ncs::sim::CUDA;
  unsigned int threads_per_block = CUDA::getThreadsPerBlock(num_neurons);
  unsigned int num_blocks = CUDA::getNumberOfBlocks(num_neurons);
  unsigned int shared_memory = threads_per_block * sizeof(ncs::sim::Bit::Word);
  updateNeuronsKernel<<<num_blocks,
                        threads_per_block,
                        shared_memory,
                        CUDA::getStream()>>>(as,
                                             bs,
                                             cs,
                                             ds,
                                             thresholds,
                                             synaptic_current,
                                             input_current,
                                             old_u,
                                             old_v,
                                             new_u,
                                             new_v,
                                             fire_vector,
                                             step_dt,
                                             num_neurons);
  return CUDA::synchronize();
}

} // namespace cuda
