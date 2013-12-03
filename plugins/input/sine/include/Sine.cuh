#include <ncs/sim/Bit.h>

void addCurrent(const float* amplitude_scale,
                const float* time_scale,
                const float* phase,
                const float* amplitude_shift,
                const unsigned int* neuron_ids,
                float* currents,
                float dt,
                unsigned int num_inputs);

void setVoltageClamp(const float* amplitude_scale,
                     const float* time_scale,
                     const float* phase,
                     const float* amplitude_shift,
                     const unsigned int* neuron_ids,
                     float* clamp_voltage_values,
                     ncs::sim::Bit::Word* voltage_clamp_bits,
                     float dt,
                     unsigned int num_inputs);
