#include <ncs/sim/FactoryMap.h>
#include <ncs/sim/NeuronSimulator.h>

extern "C" {

bool load(ncs::sim::FactoryMap<ncs::sim::NeuronSimulator>* plugin_map) {
  return true;
}
  
}
