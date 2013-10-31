#include <ncs/sim/FactoryMap.h>

#include "NCS.h"

void* createInstantiator(ncs::spec::ModelParameters* parameters) {
  return nullptr;
}

template<ncs::sim::DeviceType::Type MType>
ncs::sim::NeuronSimulator<MType>* createSimulator() {
  return new NCSSimulator<MType>();
}

extern "C" {

bool load(ncs::sim::FactoryMap<ncs::sim::NeuronSimulator>* plugin_map) {
  bool result = true;
  result &= plugin_map->registerInstantiator("ncs", createInstantiator);

  const auto CPU = ncs::sim::DeviceType::CPU;
  result &= 
    plugin_map->registerCPUProducer("ncs", createSimulator<CPU>);

#ifdef NCS_CUDA
  const auto CUDA = ncs::sim::DeviceType::CUDA;
  result &=
    plugin_map->registerCUDAProducer("ncs", createSimulator<CUDA>);
#endif // NCS_CUDA
  return result;
}
  
}
