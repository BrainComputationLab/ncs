#include <ncs/sim/FactoryMap.h>

#include "Sine.h"

#ifdef NCS_CUDA
#include "Sine.cuh"
#endif // NCS_CUDA

bool set(ncs::spec::Generator*& target,
         const std::string& parameter,
         ncs::spec::ModelParameters* parameters) {
  ncs::spec::Generator* generator = parameters->getGenerator(parameter);
  target = generator;
  if (!generator) {
    std::cerr << "sine requires " << parameter << " to be defined." <<
      std::endl;
    return false;
  }
  return true;
}

void* createInstantiator(ncs::spec::ModelParameters* parameters) {
  Instantiator* instantiator = new Instantiator();
  bool result = true;
  result &= set(instantiator->amplitude_scale, 
                "amplitude_scale", 
                parameters);
  result &= set(instantiator->time_scale, 
                "time_scale", 
                parameters);
  result &= set(instantiator->phase, 
                "phase", 
                parameters);
  result &= set(instantiator->amplitude_shift, 
                "amplitude_shift", 
                parameters);
  if (!result) {
    std::cerr << "Failed to create sine generator." << std::endl;
    delete instantiator;
    return nullptr;
  }
  return instantiator;
}

template<>
bool SineSimulator<ncs::sim::DeviceType::CPU, InputType::Voltage>::
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
    const float* amplitude_scale = batch->amplitude_scale;
    const float* time_scale = batch->time_scale;
    const float* phase = batch->phase;
    const float* amplitude_shift = batch->amplitude_shift;
    size_t count = batch->count;
    for (size_t i = 0; i < count; ++i) {
      unsigned int id = device_neuron_id[i];
      unsigned int word_index = Bit::word(id);
      Bit::Word mask = Bit::mask(id);
      clamp_bit_or.write(voltage_clamp_bits + word_index, mask);
      float amplitude =
        amplitude_scale[i] * sin(time_scale[i] * dt + phase[i]) + 
        amplitude_shift[i];
      clamp_voltage_set.write(clamp_voltage_values + id, amplitude); 
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
bool SineSimulator<ncs::sim::DeviceType::CUDA, InputType::Voltage>::
update_(ncs::sim::InputUpdateParameters* parameters) {
  float* clamp_voltage_values = parameters->clamp_voltage_values;
  ncs::sim::Bit::Word* voltage_clamp_bits = parameters->voltage_clamp_bits;
  for (auto batch : active_batches_) {
    float dt = parameters->simulation_time - batch->start_time;
    setVoltageClamp(batch->amplitude_scale,
                    batch->time_scale,
                    batch->phase,
                    batch->amplitude_shift,
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
bool SineSimulator<ncs::sim::DeviceType::CPU, InputType::Current>::
update_(ncs::sim::InputUpdateParameters* parameters) {
  ncs::sim::AtomicWriter<float> current_adder;
  float* input_current = parameters->input_current;
  std::mutex* write_lock = parameters->write_lock;
  for (auto batch : active_batches_) {
    float dt = parameters->simulation_time - batch->start_time;
    const unsigned int* device_neuron_id = batch->device_neuron_id;
    const float* amplitude_scale = batch->amplitude_scale;
    const float* time_scale = batch->time_scale;
    const float* phase = batch->phase;
    const float* amplitude_shift = batch->amplitude_shift;
    size_t count = batch->count;
    for (size_t i = 0; i < count; ++i) {
      float amplitude =
        amplitude_scale[i] * sin(time_scale[i] * dt + phase[i]) + 
        amplitude_shift[i];
      current_adder.write(input_current + device_neuron_id[i], amplitude); 
    }
  }
  std::unique_lock<std::mutex> lock(*write_lock);
  current_adder.commit(ncs::sim::AtomicWriter<float>::Add);
  lock.unlock();
  return true;
}

#ifdef NCS_CUDA

template<>
bool SineSimulator<ncs::sim::DeviceType::CUDA, InputType::Current>::
update_(ncs::sim::InputUpdateParameters* parameters) {
  float* input_current = parameters->input_current;
  for (auto batch : active_batches_) {
    float dt = parameters->simulation_time - batch->start_time;
    addCurrent(batch->amplitude_scale,
               batch->time_scale,
               batch->phase,
               batch->amplitude_shift,
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
  return new SineSimulator<MType, IType>();
}

extern "C" {

bool load(ncs::sim::FactoryMap<ncs::sim::InputSimulator>* plugin_map) {
  bool result = true;
  result &= plugin_map->registerInstantiator("sine_current",
                                             createInstantiator);
  result &= plugin_map->registerInstantiator("sine_voltage",
                                             createInstantiator);
  using ncs::sim::DeviceType;
  result &= 
    plugin_map->registerCPUProducer("sine_current",
                                    createSimulator<DeviceType::CPU,
                                                    InputType::Current>);
  result &= 
    plugin_map->registerCPUProducer("sine_voltage",
                                    createSimulator<DeviceType::CPU,
                                                    InputType::Voltage>);

#ifdef NCS_CUDA
  result &= 
    plugin_map->registerCUDAProducer("sine_current",
                                     createSimulator<DeviceType::CUDA,
                                                     InputType::Current>);
  result &= 
    plugin_map->registerCUDAProducer("sine_voltage",
                                     createSimulator<DeviceType::CUDA,
                                                     InputType::Voltage>);
#endif
  return result;
}

}
