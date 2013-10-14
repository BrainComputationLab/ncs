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

template<>
bool RectangularSimulator<ncs::sim::DeviceType::CPU, InputType::Voltage>::
update_(ncs::sim::InputUpdateParameters* parameters) {
  using ncs::sim::Bit;
  float* clamp_voltage_values = parameters->clamp_voltage_values;
  Bit::Word* voltage_clamp_bits = parameters->voltage_clamp_bits;
  std::mutex* write_lock = parameters->write_lock;
  ncs::sim::AtomicWriter<Bit::Word> clamp_bit_or;
  ncs::sim::AtomicWriter<float> clamp_voltage_set;
  for (auto batch : active_batches_) {
    const unsigned int* device_neuron_id = batch->device_neuron_id;
    const float* amplitude = batch->amplitude;
    size_t count = batch->count;
    for (size_t i = 0; i < count; ++i) {
      unsigned int id = device_neuron_id[i];
      unsigned int word_index = Bit::word(id);
      Bit::Word mask = Bit::mask(id);
      clamp_bit_or.write(voltage_clamp_bits + word_index, mask);
      clamp_voltage_set.write(clamp_voltage_values + id, amplitude[i]);
    }
  }
  std::unique_lock<std::mutex> lock(*write_lock);
  clamp_bit_or.commit(ncs::sim::AtomicWriter<Bit::Word>::Or);
  clamp_voltage_set.commit(ncs::sim::AtomicWriter<float>::Set);
  lock.unlock();
  return true;
}


template<>
bool RectangularSimulator<ncs::sim::DeviceType::CUDA, InputType::Voltage>::
update_(ncs::sim::InputUpdateParameters* parameters) {
  std::cout << "STUB: RectangularSimulator<CUDA, Voltage>::update_()" <<
    std::endl;
  return true;
}


template<>
bool RectangularSimulator<ncs::sim::DeviceType::CPU, InputType::Current>::
update_(ncs::sim::InputUpdateParameters* parameters) {
  ncs::sim::AtomicWriter<float> current_adder;
  float* input_current = parameters->input_current;
  std::mutex* write_lock = parameters->write_lock;
  for (auto batch : active_batches_) {
    const unsigned int* device_neuron_id = batch->device_neuron_id;
    const float* amplitude = batch->amplitude;
    size_t count = batch->count;
    for (size_t i = 0; i < count; ++i) {
      current_adder.write(input_current + device_neuron_id[i], amplitude[i]);
    }
  }
  std::unique_lock<std::mutex> lock(*write_lock);
  current_adder.commit(ncs::sim::AtomicWriter<float>::Add);
  lock.unlock();
  return true;
}


template<>
bool RectangularSimulator<ncs::sim::DeviceType::CUDA, InputType::Current>::
update_(ncs::sim::InputUpdateParameters* parameters) {
  std::cout << "STUB: RectangularSimulator<CUDA, Current>::update_()" <<
    std::endl;
  return true;
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
