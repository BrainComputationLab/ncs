#pragma once

#include <mutex>

#include <ncs/sim/Bit.h>

namespace ncs {

namespace sim {

struct NeuronUpdateParameters {
  const float* input_current;
  const float* clamp_voltage_values;
  const Bit::Word* voltage_clamp_bits;
  const float* previous_neuron_voltage;
  const float* synaptic_current;
  float* neuron_voltage;
  Bit::Word* neuron_fire_bits;
  std::mutex* write_lock;
};

} // namespace sim

} // namespace ncs
