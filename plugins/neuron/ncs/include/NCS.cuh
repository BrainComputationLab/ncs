#include <ncs/sim/Bit.h>

namespace cuda {

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
                   const float* calcium_persistence,
                   const float* const* spike_shape,
                   const float* threshold,
                   ncs::sim::Bit::Word* neuron_fire_bits,
                   float* new_voltage,
                   int* new_spike_shape_state,
                   float* new_calcium,
                   float* device_neuron_voltage,
                   unsigned int num_neurons);

} // namespace cuda
