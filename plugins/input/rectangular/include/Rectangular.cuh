#include <ncs/sim/Bit.h>

void addCurrent(const float* amplitudes,
                const unsigned int* neuron_ids,
                float* currents,
                unsigned int num_inputs);

void setVoltageClamp(const float* amplitudes,
                     const unsigned int* neuron_ids,
                     float* clamp_voltage_values,
                     ncs::sim::Bit::Word* voltage_clamp_bits,
                     unsigned int num_inputs);
