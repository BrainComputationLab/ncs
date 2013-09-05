#include <ncs/sim/FactoryMap.h>
#include <ncs/sim/NeuronSimulator.h>

#include "Izhikevich.h"

bool set(ncs::spec::Generator*& target,
         const std::string& parameter,
         ncs::spec::ModelParameters* parameters) {
  ncs::spec::Generator* generator = parameters->getGenerator(parameter);
  target = generator;
  if (!generator) {
    std::cerr << "izhikevich requires " << parameter << " to be defined." <<
      std::endl;
    return false;
  }
  return true;
}

void* createInstantiator(ncs::spec::ModelParameters* parameters) {
  Instantiator* instantiator = new Instantiator();
  bool result = true;
  result &= set(instantiator->a, "a", parameters);
  result &= set(instantiator->b, "b", parameters);
  result &= set(instantiator->c, "c", parameters);
  result &= set(instantiator->d, "d", parameters);
  result &= set(instantiator->u, "u", parameters);
  result &= set(instantiator->v, "v", parameters);
  if (!result) {
    std::cerr << "Failed to create izhikevich generator" << std::endl;
    delete instantiator;
    return nullptr;
  }
  return instantiator;
}

extern "C" {

bool load(ncs::sim::FactoryMap<ncs::sim::NeuronSimulator>* plugin_map) {
  bool result = true;
  result &= plugin_map->registerInstantiator("izhikevich", createInstantiator);
  return result;
}
  
}
