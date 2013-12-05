#include <ncs/cuda/CUDA.h>
#include <ncs/sim/CUDA.h>

#include "HH.cuh"

namespace cuda {

__global__ 
void resetParticleProductsKernel(float* products,
                                 unsigned int num_channels) {
  for (size_t i = grid::thread(); i < num_channels; i += grid::stride()) {
    products[i] = 1.0f;
  }
}

__device__ float solve(float a,
                       float b,
                       float c, 
                       float d, 
                       float f, 
                       float h, 
                       float v) {
  float numerator = a + b * v;
  float exponent = 0.0f;
  if (f != 0.0f) {
    exponent = (v + d) / f;
  }
  float denominator = c + h * exp(exponent);
  if (denominator == 0.0f) {
    return 0.0f;
  }
  return numerator / denominator;
}

__global__ void updateParticlesKernel(const float* alpha_a,
                                      const float* alpha_b,
                                      const float* alpha_c,
                                      const float* alpha_d,
                                      const float* alpha_f,
                                      const float* alpha_h,
                                      const float* beta_a,
                                      const float* beta_b,
                                      const float* beta_c,
                                      const float* beta_d,
                                      const float* beta_f,
                                      const float* beta_h,
                                      const float* old_x,
                                      const float* neuron_voltages,
                                      const unsigned int* neuron_id_by_particle,
                                      const float* power,
                                      const unsigned int* particle_indices,
                                      float* new_x,
                                      float* particle_products,
                                      float dt,
                                      unsigned int num_particles) {
  for (size_t i = grid::thread(); i < num_particles; i += grid::stride()) {
    float voltage = neuron_voltages[neuron_id_by_particle[i]];
    float alpha = solve(alpha_a[i],
                        alpha_b[i],
                        alpha_c[i],
                        alpha_d[i],
                        alpha_f[i],
                        alpha_h[i],
                        voltage);
    float beta = solve(beta_a[i],
                       beta_b[i],
                       beta_c[i],
                       beta_d[i],
                       beta_f[i],
                       beta_h[i],
                       voltage);
    float tau = 1.0f / (alpha + beta);
    float x_0 = alpha * tau;
    float x = old_x[i];
    float A = x_0 / tau;
    float B = 1.0f / tau;
    float e_minus_B_dt = exp(-B * dt);
    x = x * e_minus_B_dt + A / B * (1.0 - e_minus_B_dt);
    if (x < 0.0f) x = 0.0f;
    if (x > 1.0f) x = 1.0f;
    new_x[i] = x;
    particle_products[particle_indices[i]] *= pow(x, power[i]);
  }
}

__global__ void addCurrentKernel(const unsigned int* neuron_plugin_ids,
                                 const float* conductances,
                                 const float* particle_products,
                                 const float* reversal_potentials,
                                 float* channel_current,
                                 float* channel_reversal_current,
                                 unsigned int num_channels) {
  for (size_t i = grid::thread(); i < num_channels; i += grid::stride()) {
    unsigned int neuron_plugin_id = neuron_plugin_ids[i];
    float reversal_potential = reversal_potentials[i];
    float conductance = conductances[i];
    float particle_product = particle_products[i];
    float forward_current = particle_product * conductance;
    atomicAdd(channel_current + neuron_plugin_id, forward_current);
    float reversal_current = 
      particle_product * conductance * reversal_potential;
    atomicAdd(channel_reversal_current + neuron_plugin_id, reversal_current);
  }
}

bool resetParticleProducts(float* particle_products,
                           unsigned int num_channels) {
  using ncs::sim::CUDA;
  resetParticleProductsKernel<<<CUDA::getNumberOfBlocks(num_channels),
                                CUDA::getThreadsPerBlock(num_channels),
                                0,
                                CUDA::getStream()>>>(particle_products,
                                                     num_channels);
  return true;
}

__global__ void updateHHKernel(const float* old_voltage,
                               const float* input_current,
                               const float* synaptic_current,
                               const float* channel_reversal_current,
                               const float* channel_current,
                               const float* capacitances,
                               const float* thresholds,
                               ncs::sim::Bit::Word* fire_vector,
                               float* new_voltage,
                               float* device_neuron_voltage,
                               float dt,
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
      float previous_voltage = old_voltage[i];
      float external_current = input_current[i] + synaptic_current[i];
      float reversal_current = channel_reversal_current[i];
      float capacitance = capacitances[i];
      float one_over_capacitance = 1.0f / capacitance;
      float A = reversal_current + external_current * one_over_capacitance;
      float B = channel_current[i] * one_over_capacitance;
      float e_minus_B_dt = exp(-B * dt);
      float current_voltage = 
        previous_voltage * e_minus_B_dt + A / B * (1.0f - e_minus_B_dt);
      float threshold = thresholds[i];
      if (previous_voltage <= threshold && current_voltage > threshold) {
        warp_result = mask;
      }
      new_voltage[i] = current_voltage;
      device_neuron_voltage[i] = current_voltage;
    }
    warp::reduceOr(result_vector_base, warp_thread);
    if (warp::leader()) {
      fire_vector[bit::word(i)] |= warp_result;
    }
  }
}

bool updateParticles(const float* alpha_a,
                     const float* alpha_b,
                     const float* alpha_c,
                     const float* alpha_d,
                     const float* alpha_f,
                     const float* alpha_h,
                     const float* beta_a,
                     const float* beta_b,
                     const float* beta_c,
                     const float* beta_d,
                     const float* beta_f,
                     const float* beta_h,
                     const float* old_x,
                     const float* neuron_voltages,
                     const unsigned int* neuron_id_by_particle,
                     const float* power,
                     const unsigned int* particle_indices,
                     float* new_x,
                     float* particle_products,
                     float dt,
                     unsigned int num_particles) {
  using ncs::sim::CUDA;
  updateParticlesKernel<<<CUDA::getNumberOfBlocks(num_particles),
                          CUDA::getThreadsPerBlock(num_particles),
                          0,
                          CUDA::getStream()>>>(alpha_a,
                                               alpha_b,
                                               alpha_c,
                                               alpha_d,
                                               alpha_f,
                                               alpha_h,
                                               beta_a,
                                               beta_b,
                                               beta_c,
                                               beta_d,
                                               beta_f,
                                               beta_h,
                                               old_x,
                                               neuron_voltages,
                                               neuron_id_by_particle,
                                               power,
                                               particle_indices,
                                               new_x,
                                               particle_products,
                                               dt,
                                               num_particles);
  return true;
}

bool addCurrent(const unsigned int* neuron_plugin_ids,
                const float* conductances,
                const float* particle_products,
                const float* reversal_potentials,
                float* channel_current,
                float* channel_reversal_current,
                unsigned int num_channels) {
  using ncs::sim::CUDA;
  addCurrentKernel<<<CUDA::getNumberOfBlocks(num_channels),
                     CUDA::getThreadsPerBlock(num_channels),
                     0,
                     CUDA::getStream()>>>(neuron_plugin_ids,
                                          conductances,
                                          particle_products,
                                          reversal_potentials,
                                          channel_current,
                                          channel_reversal_current,
                                          num_channels); 
  return CUDA::synchronize();
}

bool updateHH(const float* old_voltage,
              const float* input_current,
              const float* synaptic_current,
              const float* channel_reversal_current,
              const float* channel_current,
              const float* capacitances,
              const float* thresholds,
              ncs::sim::Bit::Word* fire_vector,
              float* new_voltage,
              float* device_neuron_voltage,
              float dt,
              unsigned int num_neurons) {
  using ncs::sim::CUDA;
  unsigned int threads_per_block = CUDA::getThreadsPerBlock(num_neurons);
  unsigned int num_blocks = CUDA::getNumberOfBlocks(num_neurons);
  unsigned int shared_memory = threads_per_block * sizeof(ncs::sim::Bit::Word);
  updateHHKernel<<<num_blocks,
                   threads_per_block,
                   shared_memory,
                   CUDA::getStream()>>>(old_voltage,
                                        input_current,
                                        synaptic_current,
                                        channel_reversal_current,
                                        channel_current,
                                        capacitances,
                                        thresholds,
                                        fire_vector,
                                        new_voltage,
                                        device_neuron_voltage,
                                        dt,
                                        num_neurons);
  return CUDA::synchronize();
}

} // namespace cuda
