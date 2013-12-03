#include <ncs/sim/Bit.h>

void addCurrent(const float* starting_amplitudes,
                const float* slopes,
                const unsigned int* neuron_ids,
                float* currents,
                float dt,
                unsigned int num_inputs);

void setVoltageClamp(const float* starting_amplitudes,
                     const float* slopes,
                     const unsigned int* neuron_ids,
                     float* clamp_voltage_values,
                     ncs::sim::Bit::Word* voltage_clamp_bits,
                     float dt,
                     unsigned int num_inputs);
