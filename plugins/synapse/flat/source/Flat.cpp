#include <ncs/sim/FactoryMap.h>

#include "Flat.h"

bool set(ncs::spec::Generator*& target,
         const std::string& parameter,
         ncs::spec::ModelParameters* parameters) {
  ncs::spec::Generator* generator = parameters->getGenerator(parameter);
  target = generator;
  if (!generator) {
    std::cerr << "flat requires " << parameter << " to be defined." <<
      std::endl;
    return false;
  }
  return true;
}

void* createInstantiator(ncs::spec::ModelParameters* parameters) {
  Instantiator* instantiator = new Instantiator();
  bool result = true;
  result &= set(instantiator->delay, "delay", parameters);
  result &= set(instantiator->current, "current", parameters);
  if (!result) {
    std::cerr << "Failed to create flat generator" << std::endl;
    delete instantiator;
    return nullptr;
  }
  return instantiator;
}

template<ncs::sim::DeviceType::Type MType>
ncs::sim::SynapseSimulator<MType>* createSimulator() {
  return new FlatSimulator<MType>();
}

extern "C" {

bool load(ncs::sim::FactoryMap<ncs::sim::SynapseSimulator>* plugin_map) {
  bool result = true;
  result &= plugin_map->registerInstantiator("flat", createInstantiator);
  
  const auto CPU = ncs::sim::DeviceType::CPU;
  result &= plugin_map->registerCPUProducer("flat", createSimulator<CPU>);

  const auto CUDA = ncs::sim::DeviceType::CUDA;
  result &= plugin_map->registerCUDAProducer("flat", createSimulator<CUDA>);
  
  return result;
}
  
}
