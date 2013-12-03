#include <ncs/cuda/CUDA.h>
#include <ncs/sim/CUDA.h>
#include "Sine.cuh"

__global__ void addCurrentKernel(const float* amplitude_scale,
                                 const float* time_scale,
                                 const float* phase,
                                 const float* amplitude_shift,
                                 const unsigned int* neuron_ids,
                                 float* currents,
                                 float dt,
                                 unsigned int num_inputs) {
  for (unsigned int i = grid::thread(); i < num_inputs; i+= grid::stride()) {
    float amplitude =
      amplitude_scale[i] * sin(time_scale[i] * dt + phase[i]) + 
      amplitude_shift[i];
    unsigned int neuron_id = neuron_ids[i];
    atomicAdd(currents + neuron_id, amplitude);
  }
}

__global__ void setVoltageClampKernel(const float* amplitude_scale,
                                      const float* time_scale,
                                      const float* phase,
                                      const float* amplitude_shift,
                                      const unsigned int* neuron_ids,
                                      float* clamp_voltage_values,
                                      ncs::sim::Bit::Word* voltage_clamp_bits,
                                      float dt,
                                      unsigned int num_inputs) {
  for (unsigned int i = grid::thread(); i < num_inputs; i+= grid::stride()) {
    float amplitude =
      amplitude_scale[i] * sin(time_scale[i] * dt + phase[i]) + 
      amplitude_shift[i];
    unsigned int neuron_id = neuron_ids[i];
    clamp_voltage_values[i] = amplitude;
    atomicOr(voltage_clamp_bits + bit::word(neuron_id), bit::mask(neuron_id));
  }
}

void addCurrent(const float* amplitude_scale,
                const float* time_scale,
                const float* phase,
                const float* amplitude_shift,
                const unsigned int* neuron_ids,
                float* currents,
                float dt,
                unsigned int num_inputs) {
  addCurrentKernel<<<ncs::sim::CUDA::getThreadsPerBlock(num_inputs),
                     ncs::sim::CUDA::getNumberOfBlocks(num_inputs),
                     0,
                     ncs::sim::CUDA::getStream()>>>(amplitude_scale,
                                                    time_scale,
                                                    phase,
                                                    amplitude_shift,
                                                    neuron_ids,
                                                    currents,
                                                    dt,
                                                    num_inputs);
}


void setVoltageClamp(const float* amplitude_scale,
                     const float* time_scale,
                     const float* phase,
                     const float* amplitude_shift,
                     const unsigned int* neuron_ids,
                     float* clamp_voltage_values,
                     ncs::sim::Bit::Word* voltage_clamp_bits,
                     float dt,
                     unsigned int num_inputs) {
  setVoltageClampKernel<<<ncs::sim::CUDA::getThreadsPerBlock(num_inputs),
                          ncs::sim::CUDA::getNumberOfBlocks(num_inputs),
                          0,
                          ncs::sim::CUDA::getStream()>>>(amplitude_scale,
                                                         time_scale,
                                                         phase,
                                                         amplitude_shift,
                                                         neuron_ids,
                                                         clamp_voltage_values,
                                                         voltage_clamp_bits,
                                                         dt,
                                                         num_inputs);
}

