#include <ncs/cuda/CUDA.h>
#include <ncs/sim/CUDA.h>
#include "Linear.cuh"

__global__ void addCurrentKernel(const float* starting_amplitudes,
                                 const float* slopes,
                                 const unsigned int* neuron_ids,
                                 float* currents,
                                 float dt,
                                 unsigned int num_inputs) {
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int stride = blockDim.x * gridDim.x;
  for (; index < num_inputs; index += stride) {
    float amplitude = starting_amplitudes[index] + dt * slopes[index];
    unsigned int neuron_id = neuron_ids[index];
    atomicAdd(currents + neuron_id, amplitude);
  }
}

__global__ void setVoltageClampKernel(const float* starting_amplitudes,
                                      const float* slopes,
                                      const unsigned int* neuron_ids,
                                      float* clamp_voltage_values,
                                      ncs::sim::Bit::Word* voltage_clamp_bits,
                                      float dt,
                                      unsigned int num_inputs) {
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int stride = blockDim.x * gridDim.x;
  for (; index < num_inputs; index += stride) {
    float amplitude = starting_amplitudes[index] * dt * slopes[index];
    unsigned int neuron_id = neuron_ids[index];
    clamp_voltage_values[index] = amplitude;
    atomicOr(voltage_clamp_bits + bit::word(neuron_id), bit::mask(neuron_id));
  }
}

void addCurrent(const float* starting_amplitudes,
                const float* slopes,
                const unsigned int* neuron_ids,
                float* currents,
                float dt,
                unsigned int num_inputs) {
  addCurrentKernel<<<ncs::sim::CUDA::getNumberOfBlocks(num_inputs),
                     ncs::sim::CUDA::getThreadsPerBlock(num_inputs),
                     0,
                     ncs::sim::CUDA::getStream()>>>(starting_amplitudes,
                                                    slopes,
                                                    neuron_ids,
                                                    currents,
                                                    dt,
                                                    num_inputs);
}


void setVoltageClamp(const float* starting_amplitudes,
                     const float* slopes,
                     const unsigned int* neuron_ids,
                     float* clamp_voltage_values,
                     ncs::sim::Bit::Word* voltage_clamp_bits,
                     float dt,
                     unsigned int num_inputs) {
  setVoltageClampKernel<<<ncs::sim::CUDA::getNumberOfBlocks(num_inputs),
                          ncs::sim::CUDA::getThreadsPerBlock(num_inputs),
                          0,
                          ncs::sim::CUDA::getStream()>>>(starting_amplitudes,
                                                         slopes,
                                                         neuron_ids,
                                                         clamp_voltage_values,
                                                         voltage_clamp_bits,
                                                         dt,
                                                         num_inputs);
}

