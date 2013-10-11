#pragma once

#include <mutex>

#include <ncs/sim/Bit.h>

namespace ncs {

namespace sim {

struct InputUpdateParameters {
  float* input_current;
  float* clamp_voltage_values;
  Bit::Word* voltage_clamp_bits;
  std::mutex* write_lock;
  float simulation_time; // in seconds
  float time_step; // in seconds
};

} // namespace sim

} // namespace ncs
