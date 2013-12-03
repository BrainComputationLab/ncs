#include <ncs/sim/FactoryMap.h>

#include "Linear.h"

#ifdef NCS_CUDA
#include "Linear.cuh"
#endif // NCS_CUDA

bool set(ncs::spec::Generator*& target,
         const std::string& parameter,
         ncs::spec::ModelParameters* parameters) {
  ncs::spec::Generator* generator = parameters->getGenerator(parameter);
  target = generator;
  if (!generator) {
    std::cerr << "linear requires " << parameter << " to be defined." <<
      std::endl;
    return false;
  }
  return true;
}

void* createInstantiator(ncs::spec::ModelParameters* parameters) {
  Instantiator* instantiator = new Instantiator();
  bool result = true;
  result &= set(instantiator->starting_amplitude, 
                "starting_amplitude", 
                parameters);
  result &= set(instantiator->ending_amplitude, 
                "ending_amplitude", 
                parameters);
  if (!result) {
    std::cerr << "Failed to create linear generator." << std::endl;
    delete instantiator;
    return nullptr;
  }
  return instantiator;
}

template<>
bool LinearSimulator<ncs::sim::DeviceType::CPU, InputType::Voltage>::
update_(ncs::sim::InputUpdateParameters* parameters) {
  using ncs::sim::Bit;
  float* clamp_voltage_values = parameters->clamp_voltage_values;
  Bit::Word* voltage_clamp_bits = parameters->voltage_clamp_bits;
  std::mutex* write_lock = parameters->write_lock;
  ncs::sim::AtomicWriter<Bit::Word> clamp_bit_or;
  ncs::sim::AtomicWriter<float> clamp_voltage_set;
  for (auto batch : active_batches_) {
    float dt = parameters->simulation_time - batch->start_time; 
    const unsigned int* device_neuron_id = batch->device_neuron_id;
    const float* amplitude = batch->starting_amplitude;
    const float* slope = batch->slope;
    size_t count = batch->count;
    for (size_t i = 0; i < count; ++i) {
      unsigned int id = device_neuron_id[i];
      unsigned int word_index = Bit::word(id);
      Bit::Word mask = Bit::mask(id);
      clamp_bit_or.write(voltage_clamp_bits + word_index, mask);
      clamp_voltage_set.write(clamp_voltage_values + id, 
                              amplitude[i] + dt * slope[i]);
    }
  }
  std::unique_lock<std::mutex> lock(*write_lock);
  clamp_bit_or.commit(ncs::sim::AtomicWriter<Bit::Word>::Or);
  clamp_voltage_set.commit(ncs::sim::AtomicWriter<float>::Set);
  lock.unlock();
  return true;
}

#ifdef NCS_CUDA

template<>
bool LinearSimulator<ncs::sim::DeviceType::CUDA, InputType::Voltage>::
update_(ncs::sim::InputUpdateParameters* parameters) {
  float* clamp_voltage_values = parameters->clamp_voltage_values;
  ncs::sim::Bit::Word* voltage_clamp_bits = parameters->voltage_clamp_bits;
  for (auto batch : active_batches_) {
    float dt = parameters->simulation_time - batch->start_time;
    setVoltageClamp(batch->starting_amplitude,
                    batch->slope,
                    batch->device_neuron_id,
                    clamp_voltage_values,
                    voltage_clamp_bits,
                    dt,
                    batch->count);
  }
  ncs::sim::CUDA::synchronize();
  return true;
}

#endif // NCS_CUDA

template<>
bool LinearSimulator<ncs::sim::DeviceType::CPU, InputType::Current>::
update_(ncs::sim::InputUpdateParameters* parameters) {
  ncs::sim::AtomicWriter<float> current_adder;
  float* input_current = parameters->input_current;
  std::mutex* write_lock = parameters->write_lock;
  for (auto batch : active_batches_) {
    float dt = parameters->simulation_time - batch->start_time;
    const unsigned int* device_neuron_id = batch->device_neuron_id;
    const float* amplitude = batch->starting_amplitude;
    const float* slope = batch->slope;
    size_t count = batch->count;
    for (size_t i = 0; i < count; ++i) {
      current_adder.write(input_current + device_neuron_id[i], 
                          amplitude[i] + slope[i] * dt);
    }
  }
  std::unique_lock<std::mutex> lock(*write_lock);
  current_adder.commit(ncs::sim::AtomicWriter<float>::Add);
  lock.unlock();
  return true;
}

#ifdef NCS_CUDA

template<>
bool LinearSimulator<ncs::sim::DeviceType::CUDA, InputType::Current>::
update_(ncs::sim::InputUpdateParameters* parameters) {
  float* input_current = parameters->input_current;
  for (auto batch : active_batches_) {
    float dt = parameters->simulation_time - batch->start_time;
    addCurrent(batch->starting_amplitude,
               batch->slope,
               batch->device_neuron_id,
               input_current,
               dt,
               batch->count);
  }
  ncs::sim::CUDA::synchronize();
  return true;
}

#endif // NCS_CUDA

template<ncs::sim::DeviceType::Type MType, InputType IType>
ncs::sim::InputSimulator<MType>* createSimulator() {
  return new LinearSimulator<MType, IType>();
}

extern "C" {

bool load(ncs::sim::FactoryMap<ncs::sim::InputSimulator>* plugin_map) {
  bool result = true;
  result &= plugin_map->registerInstantiator("linear_current",
                                             createInstantiator);
  result &= plugin_map->registerInstantiator("linear_voltage",
                                             createInstantiator);
  using ncs::sim::DeviceType;
  result &= 
    plugin_map->registerCPUProducer("linear_current",
                                    createSimulator<DeviceType::CPU,
                                                    InputType::Current>);
  result &= 
    plugin_map->registerCPUProducer("linear_voltage",
                                    createSimulator<DeviceType::CPU,
                                                    InputType::Voltage>);

#ifdef NCS_CUDA
  result &= 
    plugin_map->registerCUDAProducer("linear_current",
                                     createSimulator<DeviceType::CUDA,
                                                     InputType::Current>);
  result &= 
    plugin_map->registerCUDAProducer("linear_voltage",
                                     createSimulator<DeviceType::CUDA,
                                                     InputType::Voltage>);
#endif
  return result;
}

}
