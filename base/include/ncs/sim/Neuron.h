#pragma once
#include <ncs/sim/Location.h>

namespace ncs {

namespace sim {

struct Neuron {
  unsigned int seed;
  void* instantiator;
  struct {
    unsigned int plugin;
    unsigned int device;
    unsigned int machine;
    unsigned int global;
  } id;
  Location location;
};

} // namespace sim

} // namespace ncs
