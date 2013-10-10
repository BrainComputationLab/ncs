#include <ncs/sim/FactoryMap.h>

#include "Rectangular.h"

bool set(ncs::spec::Generator*& target,
         const std::string& parameter,
         ncs::spec::ModelParameters* parameters) {
  ncs::spec::Generator* generator = parameters->getGenerator(parameter);
  target = generator;
  if (!generator) {
    std::cerr << "rectangular requires " << parameter << " to be defined." <<
      std::endl;
    return false;
  }
  return true;
}

void* createInstantiator(ncs::spec::ModelParameters* parameters) {
  Instantiator* instantiator = new Instantiator();
  bool result = true;
  result &= set(instantiator->amplitude, "amplitude", parameters);
  if (!result) {
    std::cerr << "Failed to create rectangular generator." << std::endl;
    delete instantiator;
    return nullptr;
  }
  return instantiator;
}

template<ncs::sim::DeviceType::Type MType, InputType IType>
ncs::sim::InputSimulator<MType>* createSimulator() {
  return new RectangularSimulator<MType, IType>();
}

extern "C" {

bool load(ncs::sim::FactoryMap<ncs::sim::InputSimulator>* plugin_map) {
  bool result = true;
  result &= plugin_map->registerInstantiator("rectangular_current",
                                             createInstantiator);
  result &= plugin_map->registerInstantiator("rectangular_voltage",
                                             createInstantiator);
  using ncs::sim::DeviceType;
  result &= 
    plugin_map->registerCPUProducer("rectangular_current",
                                    createSimulator<DeviceType::CPU,
                                                    InputType::Current>);
  result &= 
    plugin_map->registerCPUProducer("rectangular_voltage",
                                    createSimulator<DeviceType::CPU,
                                                    InputType::Voltage>);

#ifdef NCS_CUDA
  result &= 
    plugin_map->registerCUDAProducer("rectangular_current",
                                     createSimulator<DeviceType::CUDA,
                                                     InputType::Current>);
  result &= 
    plugin_map->registerCUDAProducer("rectangular_voltage",
                                     createSimulator<DeviceType::CUDA,
                                                     InputType::Voltage>);
#endif
  return result;
}

}
