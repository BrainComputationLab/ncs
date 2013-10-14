#include <ncs/sim/AtomicWriter.h>
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

template<>
bool FlatSimulator<ncs::sim::DeviceType::CPU>::
update(ncs::sim::SynapseUpdateParameters* parameters) {
  ncs::sim::AtomicWriter<float> current_adder;
  using ncs::sim::Bit;
  const Bit::Word* synaptic_fire = parameters->synaptic_fire;
  float* synaptic_current = parameters->synaptic_current;
  std::mutex* write_lock = parameters->write_lock;
  for (size_t i = 0; i < num_synapses_; ++i) {
    unsigned int fire_word_index = Bit::word(i);
    unsigned int mask = Bit::mask(i);
    if (synaptic_fire[fire_word_index] & mask) {
      current_adder.write(synaptic_current + device_neuron_device_ids_[i],
                          device_current_[i]);
    }
  }
  std::unique_lock<std::mutex> lock(*write_lock);
  current_adder.commit(ncs::sim::AtomicWriter<float>::add);
  lock.unlock();
  return true;
}


template<>
bool FlatSimulator<ncs::sim::DeviceType::CUDA>::
update(ncs::sim::SynapseUpdateParameters* parameters) {
  std::cout << "STUB: FlatSimulator<CUDA>::update()" << std::endl;
  return true;
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

#ifdef NCS_CUDA
  const auto CUDA = ncs::sim::DeviceType::CUDA;
  result &= plugin_map->registerCUDAProducer("flat", createSimulator<CUDA>);
#endif
  
  return result;
}
  
}
