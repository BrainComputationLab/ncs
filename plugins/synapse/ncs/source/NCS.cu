#include <stdio.h>

#include <ncs/cuda/CUDA.h>
#include <ncs/sim/CUDA.h>

#include "NCS.cuh"

namespace cuda {

__device__ void addCurrent(float* synaptic_current, 
                           const float* neuron_voltage,
                           float dt,
                           float psg_max,
                           float tau_postsynaptic_conductance,
                           unsigned int neuron_id,
                           float reversal_potential) {
  float f = dt / tau_postsynaptic_conductance;
  float g = psg_max * f * exp(1.0f - f);
  float current = g * (reversal_potential - neuron_voltage[neuron_id]);
  atomicAdd(synaptic_current + neuron_id, current);
}

__device__ void receiveFirings(const unsigned int* indices,
                               const float* tau_facilitations,
                               const float* tau_depressions,
                               const float* max_conductances,
                               const float* tau_ltps,
                               const float* tau_ltds,
                               const float* A_ltp_minimums,
                               const float* last_postfire_times,
                               const float* tau_postsynaptic_conductances,
                               const float* reversal_potentials,
                               const unsigned int* device_neuron_device_ids,
                               const float* neuron_voltage,
                               float* utilizations,
                               float* redistributions,
                               float* base_utilizations,
                               float* last_prefire_times,
                               float* A_ltps,
                               float* A_ltds,
                               float* synaptic_current,
                               float simulation_time,
                               unsigned int* fire_indices,
                               float* fire_times,
                               float* psg_maxes,
                               unsigned int* total_firings,
                               unsigned int* base_destination_index,
                               unsigned int num_indices) {

  // base_destination_index is a piece of shared memory
  if (block::leader()) {
    *base_destination_index = atomicAdd(total_firings, num_indices);
  }
  __syncthreads();

  if (block::thread() >= num_indices) {
    return;
  }

  unsigned int destination_index = *base_destination_index + block::thread();
  unsigned int i = indices[block::thread()];
  float pre_dt = simulation_time - last_prefire_times[i];
  last_prefire_times[i] = simulation_time;
  float base_utilization = base_utilizations[i];
  float utilization = utilizations[i];
  float tau_facilitation = tau_facilitations[i];
  float one_minus_base = 1.0f - base_utilization;
  if (tau_facilitation != 0.0f) {
    float exponent = -pre_dt / tau_facilitation;
    utilization += utilization * one_minus_base * exp(exponent);
    utilization = max(0.0f, min(utilization, 1.0f));
    utilizations[i] = utilization;
  }
  float tau_depression = tau_depressions[i];
  float redistribution = redistributions[i];
  if (tau_depression != 0.0f) {
    float exponent = -pre_dt / tau_depression;
    float coefficient = 1.0 - redistribution * one_minus_base;
    redistribution = 1.0 - coefficient * exp(exponent);
    redistribution = max(0.0f, min(redistribution, 1.0f));
    redistributions[i] = redistribution;
  }
  float psg_max = max_conductances[i] * utilization * redistribution;
  float tau_psg = tau_postsynaptic_conductances[i];
  unsigned int neuron_id = device_neuron_device_ids[i];
  float reversal_potential = reversal_potentials[i];
  addCurrent(synaptic_current,
             neuron_voltage,
             0.0f, 
             psg_max, 
             tau_psg, 
             neuron_id, 
             reversal_potential);

  float post_dt = simulation_time - last_postfire_times[i];
  float tau_ltp = tau_ltps[i];
  if (tau_ltp != 0.0f) {
    float exponent = -pre_dt / tau_ltp;
    A_ltps[i] = A_ltps[i] * exp(exponent) + A_ltp_minimums[i];
  }
  float tau_ltd = tau_ltds[i];
  if (tau_ltd != 0.0f) {
    float exponent = -post_dt / tau_ltd;
    base_utilization += A_ltds[i] * exp(exponent);
    base_utilizations[i] = base_utilization;
  }
  fire_times[destination_index] = simulation_time;
  psg_maxes[destination_index] = psg_max;
  fire_indices[destination_index] = i;
}

__global__ void checkPrefireKernel(const ncs::sim::Bit::Word* synaptic_fire,
                                   const float* tau_facilitations,
                                   const float* tau_depressions,
                                   const float* max_conductances,
                                   const float* tau_ltps,
                                   const float* tau_ltds,
                                   const float* A_ltp_minimums,
                                   const float* last_postfire_times,
                                   const float* tau_postsynaptic_conductances,
                                   const float* reversal_potentials,
                                   const unsigned int* device_neuron_device_ids,
                                   const float* neuron_voltage,
                                   float* utilizations,
                                   float* redistributions,
                                   float* base_utilizations,
                                   float* last_prefire_times,
                                   float* A_ltps,
                                   float* A_ltds,
                                   float* synaptic_current,
                                   float simulation_time,
                                   unsigned int* fire_indices,
                                   float* fire_times,
                                   float* psg_maxes,
                                   unsigned int* total_firings,
                                   unsigned int num_synapses) {
  extern __shared__ unsigned int queued_indices[];
  unsigned int* num_queued = (unsigned int*)(queued_indices + block::size());
  unsigned int* base_destination_index = num_queued + 1;
  if (block::leader()) {
    *num_queued = 0;
  }
  __syncthreads();

  unsigned int limit = math::ceiling(num_synapses, block::size());
  for (size_t i = grid::thread(); i < limit; i += grid::stride()) {
    bool fired = false;
    unsigned int local_index = 0;
    
    // Collect synaptic indices that actually fired
    if (i < num_synapses) {
      unsigned int fire_word_index = bit::word(i);
      unsigned int mask = bit::mask(i);
      if (synaptic_fire[fire_word_index] & mask) {
        fired = true;
        local_index = atomicAdd(num_queued, 1u);
        if (local_index < block::size()) {
          queued_indices[local_index] = i;
          fired = false;
        } else {
          local_index -= block::size();
        }
      }
    }
    __syncthreads();

    // If we collected too many for shared memory, ship them off to global
    if (*num_queued >= block::size()) {
      receiveFirings(queued_indices,
                     tau_facilitations,
                     tau_depressions,
                     max_conductances,
                     tau_ltps,
                     tau_ltds,
                     A_ltp_minimums,
                     last_postfire_times,
                     tau_postsynaptic_conductances,
                     reversal_potentials,
                     device_neuron_device_ids,
                     neuron_voltage,
                     utilizations,
                     redistributions,
                     base_utilizations,
                     last_prefire_times,
                     A_ltps,
                     A_ltds,
                     synaptic_current,
                     simulation_time,
                     fire_indices,
                     fire_times,
                     psg_maxes,
                     total_firings,
                     base_destination_index,
                     block::size());
      if (block::leader()) {
        *num_queued -= block::size();
      }
      __syncthreads();
    }
    if (fired) {
      queued_indices[local_index] = i;
    }
  }
  __syncthreads();
  
  // Handle any leftover indices in shared memory
  if (*num_queued > 0) {
      receiveFirings(queued_indices,
                     tau_facilitations,
                     tau_depressions,
                     max_conductances,
                     tau_ltps,
                     tau_ltds,
                     A_ltp_minimums,
                     last_postfire_times,
                     tau_postsynaptic_conductances,
                     reversal_potentials,
                     device_neuron_device_ids,
                     neuron_voltage,
                     utilizations,
                     redistributions,
                     base_utilizations,
                     last_prefire_times,
                     A_ltps,
                     A_ltds,
                     synaptic_current,
                     simulation_time,
                     fire_indices,
                     fire_times,
                     psg_maxes,
                     total_firings,
                     base_destination_index,
                     *num_queued);
  }
}

__device__ void computePositiveLearning(const unsigned int* indices,
                                        const float* last_prefire_times,
                                        const float* tau_ltps,
                                        const float* tau_ltds,
                                        const float* A_ltps,
                                        const float* A_ltd_minimums,
                                        float* A_ltds,
                                        float* base_utilizations,
                                        float* last_postfire_times,
                                        float simulation_time,
                                        unsigned int num_indices) {
  if (block::thread() >= num_indices) {
    return;
  }
  unsigned int i = block::thread();
  float post_dt = simulation_time - last_postfire_times[i];
  last_postfire_times[i] = simulation_time;
  float pre_dt = simulation_time - last_prefire_times[i];
  float tau_ltd = tau_ltds[i];
  if (tau_ltd != 0.0f) {
    float exponent = -post_dt / tau_ltd;
    A_ltds[i] = A_ltds[i] * exp(exponent) + A_ltd_minimums[i];
  }
  float tau_ltp = tau_ltps[i];
  if (tau_ltp != 0.0f) {
    float exponent = -pre_dt / tau_ltp;
    float base_utilization = base_utilizations[i];
    base_utilization += A_ltps[i] * exp(exponent);
    base_utilizations[i] = base_utilization;
  }
}

__global__
void checkPostfireKernel(const ncs::sim::Bit::Word* neuron_fire,
                         const unsigned int* device_neuron_device_ids,
                         const float* last_prefire_times,
                         const float* tau_ltps,
                         const float* tau_ltds,
                         const float* A_ltps,
                         const float* A_ltd_minimums,
                         float* A_ltds,
                         float* base_utilizations,
                         float* last_postfire_times,
                         float simulation_time,
                         unsigned int num_synapses) {
  extern __shared__ unsigned int queued_indices[];
  unsigned int* num_queued = (unsigned int*)(queued_indices + block::size());
  if (block::leader()) {
    *num_queued = 0;
  }
  __syncthreads();
  unsigned int limit = math::ceiling(num_synapses, block::size());
  for (size_t i = grid::thread(); i < limit; i += grid::stride()) {
    bool fired = false;
    unsigned int local_index = 0;
    if (i < num_synapses) {
      unsigned int neuron_id = device_neuron_device_ids[i];
      unsigned int fire_word_index = bit::word(neuron_id);
      unsigned int mask = bit::mask(neuron_id);
      if (neuron_fire[fire_word_index] & mask) {
        fired = true;
        local_index = atomicAdd(num_queued, 1u);
        if (local_index < block::size()) {
          queued_indices[local_index] = i;
          fired = false;
        } else {
          local_index -= block::size();
        }
      }
    }
    __syncthreads();
    if (*num_queued >= block::size()) {
      computePositiveLearning(queued_indices,
                              last_prefire_times,
                              tau_ltps,
                              tau_ltds,
                              A_ltps,
                              A_ltd_minimums,
                              A_ltds,
                              base_utilizations,
                              last_postfire_times,
                              simulation_time,
                              block::size());
      if (block::leader()) {
        *num_queued -= block::size();
      }
      __syncthreads();
    }
    if (fired) {
      queued_indices[local_index] = i;
    }
  }
  __syncthreads();
  if (*num_queued > 0) {
    computePositiveLearning(queued_indices,
                            last_prefire_times,
                            tau_ltps,
                            tau_ltds,
                            A_ltps,
                            A_ltd_minimums,
                            A_ltds,
                            base_utilizations,
                            last_postfire_times,
                            simulation_time,
                            *num_queued);
  }
}

__device__ 
void addOldFiringsToGlobal(const unsigned int* old_fire_indices,
                           const float* old_fire_times,
                           const float* old_psg_maxes,
                           const float* tau_postsynaptic_conductances,
                           const unsigned int* device_neuron_device_ids,
                           const float* reversal_potentials,
                           unsigned int* new_fire_indices,
                           float* new_fire_times,
                           float* new_psg_maxes,
                           unsigned int* total_firings,
                           float* synaptic_current,
                           const float* neuron_voltage,
                           unsigned int* base_destination_index,
                           float simulation_time,
                           unsigned int num_indices) {
  if (block::leader()) {
    *base_destination_index = atomicAdd(total_firings, num_indices);
  }
  __syncthreads();
  
  if (block::thread() >= num_indices) {
    return;
  }

  unsigned int destination_index = *base_destination_index + block::thread();
  unsigned int fire_index = old_fire_indices[block::thread()];
  float psg_max = old_psg_maxes[block::thread()];
  float fire_time = old_fire_times[block::thread()];
  float tau_psg = tau_postsynaptic_conductances[fire_index];
  unsigned int neuron_id = device_neuron_device_ids[fire_index];
  float reversal_potential = reversal_potentials[fire_index];
  float dt = simulation_time - fire_time;
  addCurrent(synaptic_current,
             neuron_voltage,
             dt, 
             psg_max, 
             tau_psg, 
             neuron_id, 
             reversal_potential);
  new_fire_indices[destination_index] = fire_index;
  new_fire_times[destination_index] = fire_time;
  new_psg_maxes[destination_index] = psg_max;
}

__global__ 
void addOldFiringsKernel(const unsigned int* old_fire_indices,
                         const float* old_fire_times,
                         const float* psg_waveform_durations,
                         const float* old_psg_maxes,
                         const float* tau_postsynaptic_conductances,
                         const unsigned int* device_neuron_device_ids,
                         const float* reversal_potentials,
                         unsigned int* new_fire_indices,
                         float* new_fire_times,
                         float* new_psg_maxes,
                         unsigned int* total_firings,
                         float* synaptic_current,
                         const float* neuron_voltage,
                         float simulation_time,
                         unsigned int num_old_firings) {
  extern __shared__ float queued_fire_times[];
  float* queued_psg_maxes = queued_fire_times + block::size();
  unsigned int* queued_fire_indices = 
    (unsigned int*)(queued_psg_maxes + block::size());
  unsigned int* num_queued = queued_fire_indices + block::size();
  unsigned int* base_destination_index = num_queued + 1;
  if (block::leader()) {
    *num_queued = 0;
  }
  __syncthreads();

  unsigned int limit = math::ceiling(num_old_firings, block::size());
  for (size_t i = grid::thread(); i < limit; i += grid::stride()) {
    bool save = false;
    unsigned int local_index = 0;
    float fire_time = 0.0f;
    float psg_max = 0.0f;
    unsigned int fire_index = 0;
    if (i < num_old_firings) {
      float fire_time = old_fire_times[i];
      unsigned int fire_index = old_fire_indices[i];
      float psg_waveform_duration = psg_waveform_durations[fire_index];
      float dt = simulation_time - fire_time;
      if (dt <= psg_waveform_duration) {
        save = true;
        local_index = atomicAdd(num_queued, 1u);
        psg_max = old_psg_maxes[i];
        if (local_index < block::size()) {
          queued_fire_indices[local_index] = fire_index;
          queued_fire_times[local_index] = fire_time;
          queued_psg_maxes[local_index] = psg_max;
          save = false;
        } else {
          local_index -= block::size();
        }
      }
    }
    __syncthreads();
    if (*num_queued >= block::size()) {
      addOldFiringsToGlobal(queued_fire_indices,
                            queued_fire_times,
                            queued_psg_maxes,
                            tau_postsynaptic_conductances,
                            device_neuron_device_ids,
                            reversal_potentials,
                            new_fire_indices,
                            new_fire_times,
                            new_psg_maxes,
                            total_firings,
                            synaptic_current,
                            neuron_voltage,
                            base_destination_index,
                            simulation_time,
                            block::size());
      if (block::leader()) {
        *num_queued -= block::size();
      }
    }
    __syncthreads();
    if (save) {
      queued_fire_indices[local_index] = fire_index;
      queued_fire_times[local_index] = fire_time;
      queued_psg_maxes[local_index] = psg_max;
    }
  }
  __syncthreads();
  if (*num_queued > 0) {
    addOldFiringsToGlobal(queued_fire_indices,
                          queued_fire_times,
                          queued_psg_maxes,
                          tau_postsynaptic_conductances,
                          device_neuron_device_ids,
                          reversal_potentials,
                          new_fire_indices,
                          new_fire_times,
                          new_psg_maxes,
                          total_firings,
                          synaptic_current,
                          neuron_voltage,
                          base_destination_index,
                          simulation_time,
                          *num_queued);
  }
}

void checkPrefire(const ncs::sim::Bit::Word* synaptic_fire,
                  const float* tau_facilitations,
                  const float* tau_depressions,
                  const float* max_conductances,
                  const float* tau_ltps,
                  const float* tau_ltds,
                  const float* A_ltp_minimums,
                  const float* last_postfire_times,
                  const float* tau_postsynaptic_conductances,
                  const float* reversal_potentials,
                  const unsigned int* device_neuron_device_ids,
                  const float* neuron_voltage,
                  float* utilizations,
                  float* redistributions,
                  float* base_utilizations,
                  float* last_prefire_times,
                  float* A_ltps,
                  float* A_ltds,
                  float* synaptic_current,
                  float simulation_time,
                  unsigned int* fire_indices,
                  float* fire_times,
                  float* psg_maxes,
                  unsigned int* total_firings,
                  unsigned int num_synapses) {
  using ncs::sim::CUDA;
  unsigned int threads_per_block = CUDA::getThreadsPerBlock(num_synapses);
  unsigned int num_blocks = CUDA::getNumberOfBlocks(num_synapses);
  unsigned int shared_memory =
    sizeof(unsigned int) * threads_per_block + // queued_indices
    sizeof(unsigned int) + // num_queued
    sizeof(unsigned int); // base_destination_index
  checkPrefireKernel<<<num_blocks,
                       threads_per_block,
                       shared_memory,
                       CUDA::getStream()>>>(synaptic_fire,
                                            tau_facilitations,
                                            tau_depressions,
                                            max_conductances,
                                            tau_ltps,
                                            tau_ltds,
                                            A_ltp_minimums,
                                            last_postfire_times,
                                            tau_postsynaptic_conductances,
                                            reversal_potentials,
                                            device_neuron_device_ids,
                                            neuron_voltage,
                                            utilizations,
                                            redistributions,
                                            base_utilizations,
                                            last_prefire_times,
                                            A_ltps,
                                            A_ltds,
                                            synaptic_current,
                                            simulation_time,
                                            fire_indices,
                                            fire_times,
                                            psg_maxes,
                                            total_firings,
                                            num_synapses);
}

void addOldFirings(const unsigned int* old_fire_indices,
                   const float* old_fire_times,
                   const float* psg_waveform_durations,
                   const float* old_psg_maxes,
                   const float* tau_postsynaptic_conductances,
                   const unsigned int* device_neuron_device_ids,
                   const float* reversal_potentials,
                   unsigned int* new_fire_indices,
                   float* new_fire_times,
                   float* new_psg_maxes,
                   unsigned int* total_firings,
                   float* synaptic_current,
                   const float* neuron_voltage,
                   float simulation_time,
                   unsigned int num_old_firings) {
  using ncs::sim::CUDA;
  unsigned int threads_per_block = CUDA::getThreadsPerBlock(num_old_firings);
  unsigned int num_blocks = CUDA::getNumberOfBlocks(num_old_firings);
  unsigned int shared_memory =
    sizeof(float) * threads_per_block + // queued_fire_times
    sizeof(float) * threads_per_block + // queued_psg_maxes
    sizeof(unsigned int) * threads_per_block + // queued_fire_indices
    sizeof(unsigned int) + // num_queued
    sizeof(unsigned int); // base_destination_index
  addOldFiringsKernel<<<num_blocks,
                        threads_per_block,
                        shared_memory,
                        CUDA::getStream()>>>(old_fire_indices,
                                             old_fire_times,
                                             psg_waveform_durations,
                                             old_psg_maxes,
                                             tau_postsynaptic_conductances,
                                             device_neuron_device_ids,
                                             reversal_potentials,
                                             new_fire_indices,
                                             new_fire_times,
                                             new_psg_maxes,
                                             total_firings,
                                             synaptic_current,
                                             neuron_voltage,
                                             simulation_time,
                                             num_old_firings);
}

void checkPostfire(const ncs::sim::Bit::Word* neuron_fire,
                   const unsigned int* device_neuron_device_ids,
                   const float* last_prefire_times,
                   const float* tau_ltps,
                   const float* tau_ltds,
                   const float* A_ltps,
                   const float* A_ltd_minimums,
                   float* A_ltds,
                   float* base_utilizations,
                   float* last_postfire_times,
                   float simulation_time,
                   unsigned int num_synapses) {
  using ncs::sim::CUDA;
  unsigned int threads_per_block = CUDA::getThreadsPerBlock(num_synapses);
  unsigned int num_blocks = CUDA::getNumberOfBlocks(num_synapses);
  unsigned int shared_memory =
    sizeof(unsigned int) * threads_per_block + // queued_indices
    sizeof(unsigned int);// num_queued
  checkPostfireKernel<<<num_blocks,
                        threads_per_block,
                        shared_memory,
                        CUDA::getStream()>>>(neuron_fire,
                                             device_neuron_device_ids,
                                             last_prefire_times,
                                             tau_ltps,
                                             tau_ltds,
                                             A_ltps,
                                             A_ltd_minimums,
                                             A_ltds,
                                             base_utilizations,
                                             last_postfire_times,
                                             simulation_time,
                                             num_synapses);
}
} // namespace cuda
