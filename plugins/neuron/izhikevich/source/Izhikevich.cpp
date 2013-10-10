#include <ncs/sim/FactoryMap.h>

#include "Izhikevich.h"

template<>
bool IzhikevichSimulator<ncs::sim::DeviceType::CPU>::
update(ncs::sim::NeuronUpdateParameters* parameters) {
  using ncs::sim::Bit;
  const float* input_current = parameters->input_current;
  const float* clamp_voltage_values = parameters->clamp_voltage_values;
  const Bit::Word* voltage_clamp_bits = parameters->voltage_clamp_bits;
  const float* previous_neuron_voltage = parameters->previous_neuron_voltage;
  const float* synaptic_current = parameters->synaptic_current;
  float* neuron_voltage = parameters->neuron_voltage;
  Bit::Word* neuron_fire_bits = parameters->neuron_fire_bits;

  // Zero out all the fire bits for our neurons
  unsigned int num_words = Bit::num_words(num_neurons_);
  for (unsigned int i = 0; i < num_words; ++i) {
    neuron_fire_bits[i] = 0;
  }
  auto dvdt = [](float v, float u, float current) {
    return 0.04f * v * v + 5.0f * v + 140.0f - u + current;
  };
  auto dudt = [](float v, float u, float a, float b) {
    return a * (b * v - u);
  };
  for (unsigned int i = 0; i < num_neurons_; ++i) {
    float a = a_[i];
    float b = b_[i];
    float c = c_[i];
    float d = d_[i];
    float u = u_[i];
    float v = previous_neuron_voltage[i];
    float threshold = threshold_[i];
    float current = input_current[i] + synaptic_current[i];
    if (v >= threshold) {
      v = c;
      u += d;
      unsigned int word_index = Bit::word(i);
      Bit::Word mask = Bit::mask(i);
      neuron_fire_bits[word_index] |= mask;
    }
    float new_v = v + step_dt_ * dvdt(v, u, current);
    float new_u = u + step_dt_ * dudt(v, u, a, b);
    v = new_v;
    u = new_u;
    new_v = v + step_dt_ * dvdt(v, u, current);
    new_u = u + step_dt_ * dudt(v, u, a, b);
    v = new_v;
    u = new_u;
    if (v >= threshold) {
      v = threshold;
    }
    neuron_voltage[i] = v;
    u_[i] = u;
  }
  return true;
}

template<>
bool IzhikevichSimulator<ncs::sim::DeviceType::CUDA>::
update(ncs::sim::NeuronUpdateParameters* parameters) {
  std::clog << "STUB: IzhikevichSimulator<CUDA>::update()" << std::endl;
  return true;
}

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
  result &= set(instantiator->threshold, "threshold", parameters);
  if (!result) {
    std::cerr << "Failed to create izhikevich generator" << std::endl;
    delete instantiator;
    return nullptr;
  }
  return instantiator;
}

template<ncs::sim::DeviceType::Type MType>
ncs::sim::NeuronSimulator<MType>* createSimulator() {
  return new IzhikevichSimulator<MType>();
}

extern "C" {

bool load(ncs::sim::FactoryMap<ncs::sim::NeuronSimulator>* plugin_map) {
  bool result = true;
  result &= plugin_map->registerInstantiator("izhikevich", createInstantiator);

  const auto CPU = ncs::sim::DeviceType::CPU;
  result &= 
    plugin_map->registerCPUProducer("izhikevich", createSimulator<CPU>);

  const auto CUDA = ncs::sim::DeviceType::CUDA;
  result &=
    plugin_map->registerCUDAProducer("izhikevich", createSimulator<CUDA>);
  return result;
}
  
}
