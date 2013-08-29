#pragma once

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
  struct {
    unsigned int plugin;
    unsigned int device;
    unsigned int machine;
  } location;
};

} // namespace sim

} // namespace ncs
