#include <ncs/sim/Bit.h>

namespace cuda {

bool resetParticleProducts(float* particle_products,
                           unsigned int num_channels);

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
                     unsigned int num_particles);

bool addCurrent(const unsigned int* neuron_plugin_ids,
                const float* conductances,
                const float* particle_products,
                const float* reversal_potentials,
                float* channel_current,
                float* channel_reversal_current,
                unsigned int num_channels);

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
              unsigned int num_neurons);

} // namespace cuda
