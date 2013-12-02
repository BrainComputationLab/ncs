#include <ncs/cuda/CUDA.h>
#include <ncs/sim/CUDA.h>

#include "NCS.cuh"

namespace cuda {

__global__ void updateNeuronsKernel(const int* old_spike_shape_state,
                                    const float* old_voltage,
                                    const float* old_calcium,
                                    const float* input_current,
                                    const float* synaptic_current,
                                    const float* channel_current,
                                    const float* resting_potential,
                                    const float* voltage_persistence,
                                    const float* dt_over_capacitance,
                                    const unsigned int* spike_shape_length,
                                    const float* calcium_spike_increment,
                                    const float* const* spike_shape,
                                    const float* threshold,
                                    ncs::sim::Bit::Word* neuron_fire_bits,
                                    float* new_voltage,
                                    int* new_spike_shape_state,
                                    float* new_calcium,
                                    float* device_neuron_voltage,
                                    unsigned int num_neurons) {
	extern __shared__ ncs::sim::Bit::Word shared_fire_vector[];
	unsigned int& warp_result = shared_fire_vector[threadIdx.x];
	unsigned int* result_vector_base = shared_fire_vector + warp::index() * 32;
	unsigned int warp_thread = warp::thread();
	unsigned int limit = (num_neurons + 31) / 32;
	unsigned int mask = bit::mask(warp_thread);
  for (size_t i = grid::thread(); i < limit; i += grid::stride()) {
    warp_result = 0;
    if (i < num_neurons) {
      int spike_shape_state = old_spike_shape_state[i];
      float voltage = old_voltage[i];
      float calcium = old_calcium[i];
      float total_current = input_current[i] + 
                            synaptic_current[i] + channel_current[i];
      if (spike_shape_state < 0) { // Do real computations
        float vm_rest = resting_potential[i];
        float dv = voltage - vm_rest;
        voltage = vm_rest + 
                  dv * voltage_persistence[i] + 
                  dt_over_capacitance[i] * total_current;
        if (voltage > threshold[i]) {
          spike_shape_state = spike_shape_length[i] - 1;
          calcium += calcium_spike_increment[i];
          warp_result = mask;
        }
      }
      if (spike_shape_state >= 0) { // Still following spike shape
        voltage = spike_shape[i][spike_shape_state];
        spike_shape_state--;
      }
      new_voltage[i] = voltage;
      new_spike_shape_state[i] = spike_shape_state;
      new_calcium[i] = calcium;
      device_neuron_voltage[i] = voltage;
    }
    warp::reduceOr(result_vector_base, warp_thread);
    if (warp::leader()) {
      neuron_fire_bits[bit::word(i)] = warp_result;
    }
  }
}

void updateNeurons(const int* old_spike_shape_state,
                   const float* old_voltage,
                   const float* old_calcium,
                   const float* input_current,
                   const float* synaptic_current,
                   const float* channel_current,
                   const float* resting_potential,
                   const float* voltage_persistence,
                   const float* dt_over_capacitance,
                   const unsigned int* spike_shape_length,
                   const float* calcium_spike_increment,
                   const float* const* spike_shape,
                   const float* threshold,
                   ncs::sim::Bit::Word* neuron_fire_bits,
                   float* new_voltage,
                   int* new_spike_shape_state,
                   float* new_calcium,
                   float* device_neuron_voltage,
                   unsigned int num_neurons) {
  unsigned int threads_per_block = 
    ncs::sim::CUDA::getThreadsPerBlock(num_neurons);
  unsigned int num_blocks = ncs::sim::CUDA::getNumberOfBlocks(num_neurons);
  unsigned int shared_memory_size = 
    sizeof(ncs::sim::Bit::Word) * threads_per_block;
  updateNeuronsKernel<<<threads_per_block,
                        num_blocks,
                        shared_memory_size,
                        ncs::sim::CUDA::getStream()>>>(old_spike_shape_state,
                                                       old_voltage,
                                                       old_calcium,
                                                       input_current,
                                                       synaptic_current,
                                                       channel_current,
                                                       resting_potential,
                                                       voltage_persistence,
                                                       dt_over_capacitance,
                                                       spike_shape_length,
                                                       calcium_spike_increment,
                                                       spike_shape,
                                                       threshold,
                                                       neuron_fire_bits,
                                                       new_voltage,
                                                       new_spike_shape_state,
                                                       new_calcium,
                                                       device_neuron_voltage,
                                                       num_neurons);
  ncs::sim::CUDA::synchronize();
}

} // namespace cuda
