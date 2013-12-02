#include <ncs/cuda/CUDA.h>
#include <ncs/sim/CUDA.h>

#include "Channel.cuh"

namespace cuda {

__global__ 
void updateVoltageGatedIonKernel(const unsigned int* neuron_plugin_ids,
                                 const float* neuron_voltages,
                                 const float* v_half,
                                 const float* deactivation_scale,
                                 const float* activation_scale,
                                 const float* equilibrium_scale,
                                 const float* tau_scale_factor,
                                 const float* old_m,
                                 const float* reversal_potential,
                                 const float* conductance,
                                 float* new_m,
                                 float* channel_current,
                                 float dt,
                                 unsigned int num_channels) {
  for (size_t i = grid::thread(); i < num_channels; i += grid::stride()) {
    unsigned int neuron_plugin_id = neuron_plugin_ids[i];
    float neuron_voltage = neuron_voltages[neuron_plugin_id];
    float d_v = neuron_voltage - v_half[i];
    float beta = exp(d_v * deactivation_scale[i]);
    float alpha = exp(d_v * activation_scale[i]);
    float tau_m = tau_scale_factor[i] / (alpha + beta);
    float m_infinity = 1.0f / (1.0f + exp(-d_v * equilibrium_scale[i]));
    float dt_over_tau = dt / tau_m;
    float m = old_m[i] * (1.0f - dt_over_tau) + m_infinity * dt_over_tau;
    m = max(0.0f, min(m, 1.0f));
    new_m[i] = m;
    float current = 
      conductance[i] * m * (reversal_potential[i] - neuron_voltage);
    atomicAdd(channel_current + neuron_plugin_id, current);
  }
}

bool updateVoltageGatedIon(const unsigned int* neuron_plugin_ids,
                           const float* neuron_voltages,
                           const float* v_half,
                           const float* deactivation_scale,
                           const float* activation_scale,
                           const float* equilibrium_scale,
                           const float* tau_scale_factor,
                           const float* old_m,
                           const float* reversal_potential,
                           const float* conductance,
                           float* new_m,
                           float* channel_current,
                           float dt,
                           unsigned int num_channels) {
  using ncs::sim::CUDA;
  updateVoltageGatedIonKernel<<<CUDA::getThreadsPerBlock(num_channels),
                                CUDA::getNumberOfBlocks(num_channels),
                                0,
                                CUDA::getStream()>>>(neuron_plugin_ids,
                                                     neuron_voltages,
                                                     v_half,
                                                     deactivation_scale,
                                                     activation_scale,
                                                     equilibrium_scale,
                                                     tau_scale_factor,
                                                     old_m,
                                                     reversal_potential,
                                                     conductance,
                                                     new_m,
                                                     channel_current,
                                                     dt,
                                                     num_channels);
  return CUDA::synchronize();
}

} // namespace cuda
