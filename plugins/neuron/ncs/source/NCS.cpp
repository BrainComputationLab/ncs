#include <ncs/sim/FactoryMap.h>

#include "NCS.h"

bool set(ncs::spec::Generator*& target,
         const std::string& parameter,
         ncs::spec::ModelParameters* parameters,
         const std::string& type) {
  ncs::spec::Generator* generator = parameters->getGenerator(parameter);
  target = generator;
  if (!generator) {
    std::cerr << type << " requires " << parameter << " to be defined." <<
      std::endl;
    return false;
  }
  return true;
}

ParticleConstantsInstantiator*
createParticleConstantsInstantiator(ncs::spec::ModelParameters* parameters) {
  ParticleConstantsInstantiator* pci = new ParticleConstantsInstantiator();
  bool result = true;
  result &= set(pci->a, "a", parameters, "particle_parameters");
  result &= set(pci->b, "b", parameters, "particle_parameters");
  result &= set(pci->c, "c", parameters, "particle_parameters");
  result &= set(pci->d, "d", parameters, "particle_parameters");
  result &= set(pci->f, "f", parameters, "particle_parameters");
  result &= set(pci->h, "h", parameters, "particle_parameters");
  if (!result) {
    std::cerr << "Failed to instantiate ParticleConstantsInstantiator." <<
      std::endl;
    delete pci;
    return nullptr;
  }
  return pci;
}

VoltageParticleInstantiator*
createVoltageParticleInstantiator(ncs::spec::ModelParameters* parameters) {
  auto alpha_generator = parameters->getGenerator("alpha");
  auto beta_generator = parameters->getGenerator("beta");
  if (nullptr == alpha_generator || nullptr == beta_generator) {
    std::cerr << "particle must define alpha and beta values." << std::endl;
    return nullptr;
  }
  ncs::spec::RNG rng(0);
  auto alpha_parameters = alpha_generator->generateParameters(&rng);
  auto beta_parameters = beta_generator->generateParameters(&rng);
  auto alpha_particle = createParticleConstantsInstantiator(alpha_parameters);
  auto beta_particle = createParticleConstantsInstantiator(beta_parameters);
  if (nullptr == alpha_particle || nullptr == beta_particle) {
    std::cerr << "Failed to instantiate alpha or beta particle." << std::endl;
    if (alpha_particle) {
      delete alpha_particle;
    }
    if (beta_particle) {
      delete beta_particle;
    }
    return nullptr;
  }
  VoltageParticleInstantiator* vpi = new VoltageParticleInstantiator();
  bool result = true;
  result &= set(vpi->power, "power", parameters, "voltage_gated particle");
  result &= set(vpi->x_initial, 
                "x_initial", 
                parameters, 
                "voltage_gated particle");
  if (!result) {
    std::cerr << "Failed to instantiate voltage_gated particle." << std::endl;
    delete vpi;
    return nullptr;
  }
  vpi->alpha = alpha_particle;
  vpi->beta = beta_particle;
  return vpi;
}

VoltageGatedInstantiator*
createVoltageGatedInstantiator(ncs::spec::ModelParameters* parameters) {
  VoltageGatedInstantiator* vgi = new VoltageGatedInstantiator();
  bool result = true;
  result &= set(vgi->conductance, "conductance", parameters, "voltage_gated");
  auto particle_generator = parameters->getGenerator("particles");
  if (nullptr == particle_generator) {
    std::cerr << "voltage_gated channels require at least one " <<
      "particle." << std::endl;
    delete vgi;
    return nullptr;
  }
  ncs::spec::RNG rng(0);
  auto particle_list = particle_generator->generateList(&rng);
  if (particle_list.empty()) {
    std::cerr << "voltage_gated channels require at least one " <<
      "particle." << std::endl;
    delete vgi;
    return nullptr;
  }
  for (auto particle_spec : particle_list) {
    auto particle_parameters = particle_spec->generateParameters(&rng);
    auto particle = createVoltageParticleInstantiator(particle_parameters);
    if (nullptr == particle) {
      std::cerr << "Failed to create a voltage-gated particle." << std::endl;
      delete vgi;
      return nullptr;
    }
    vgi->particles.push_back(particle);
  }
  return vgi;
}

ChannelInstantiator* 
createChannelInstantiator(ncs::spec::ModelParameters* parameters) {
   if (parameters->getType() == "voltage_gated") {
     return createVoltageGatedInstantiator(parameters);
   } else if (parameters->getType() == "calcium_gated") {
     // TODO (rvhoang):
   } else {
     std::cerr << "Unknown channel type " << parameters->getType() <<
       std::endl;
     return nullptr;
   }
}

void* createInstantiator(ncs::spec::ModelParameters* parameters) {
  NeuronInstantiator* instantiator = new NeuronInstantiator();
  bool result = true;
  result &= set(instantiator->threshold,
                "threshold",
                parameters,
                "ncs neuron");
  result &= set(instantiator->resting_potential,
                "resting_potential", 
                parameters,
                "ncs neuron");
  result &= set(instantiator->calcium, 
                "calcium", 
                parameters,
                "ncs neuron");
  result &= set(instantiator->calcium_spike_increment, 
                "calcium_spike_increment", 
                parameters,
                "ncs neuron");
  result &= set(instantiator->tau_calcium, 
                "tau_calcium", 
                parameters,
                "ncs neuron");
  result &= set(instantiator->leak_reversal_potential, 
                "leak_reversal_potential", 
                parameters,
                "ncs neuron");
  result &= set(instantiator->leak_conductance, 
                "leak_conductance", 
                parameters,
                "ncs neuron");
  result &= set(instantiator->tau_membrane, 
                "tau_membrane", 
                parameters,
                "ncs neuron");
  result &= set(instantiator->r_membrane, 
                "r_membrane", 
                parameters,
                "ncs neuron");
  if (!result) {
    std::cerr << "Failed to build ncs instantiator." << std::endl;
    delete instantiator;
    return nullptr;
  }
  auto channel_generator = parameters->getGenerator("channels");
  if (nullptr != channel_generator) {
    ncs::spec::RNG rng(0);
    auto channel_list = channel_generator->generateList(&rng);
    for (auto channel_generator : channel_list) {
      auto channel_parameters = channel_generator->generateParameters(&rng);
      ChannelInstantiator* ci = createChannelInstantiator(channel_parameters);
      if (nullptr == ci) {
        std::cerr << "Failed to generate channel." << std::endl;
        delete instantiator;
        return false;
      }
      instantiator->channels.push_back(ci);
    }
  }
  return instantiator;
}

template<ncs::sim::DeviceType::Type MType>
ncs::sim::NeuronSimulator<MType>* createSimulator() {
  return new NCSSimulator<MType>();
}

template<>
bool NCSSimulator<ncs::sim::DeviceType::CPU>::
update(ncs::sim::NeuronUpdateParameters* parameters) {
  using ncs::sim::Bit;
  ncs::sim::Mailbox mailbox;
  ChannelCurrentBuffer<ncs::sim::DeviceType::CPU>* channel_buffer = nullptr;
  channel_current_subscription_->pull(&channel_buffer, &mailbox);
  NeuronBuffer<ncs::sim::DeviceType::CPU>* state_buffer = nullptr;
  state_subscription_->pull(&state_buffer, &mailbox);
  if (!mailbox.wait(&channel_buffer, &state_buffer)) {
    return true;
  }
  auto blank = this->getBlank();
  const float* channel_current = channel_buffer->getCurrent();
  const float* old_voltage = state_buffer->getVoltage();
  const float* old_calcium = state_buffer->getCalcium();
  const float* input_current = parameters->input_current;
  const float* clamp_voltage_values = parameters->clamp_voltage_values;
  const Bit::Word* voltage_clamp_bits = parameters->voltage_clamp_bits;
  const float* synaptic_current = parameters->synaptic_current;
  float* device_neuron_voltage = parameters->neuron_voltage;
  float* new_calcium = blank->getCalcium();
  float* new_voltage = blank->getVoltage();
  Bit::Word* neuron_fire_bits = parameters->neuron_fire_bits;
  unsigned int num_words = Bit::num_words(num_neurons_);
  for (unsigned int i = 0; i < num_words; ++i) {
    neuron_fire_bits[i] = 0;
  }
  for (unsigned int i = 0; i < num_neurons_; ++i) {
    float previous_voltage = old_voltage[i];
    float total_current = input_current[i] +
                          synaptic_current[i] + 
                          channel_current[i];
    total_current -= leak_conductance_[i] * 
                     (previous_voltage - leak_reversal_potential_[i]);
    float resting_voltage = resting_potential_[i];
    float dv = previous_voltage - resting_voltage;
    float current_voltage = resting_voltage + 
                        dv * voltage_persistence_[i] + 
                        total_current * dt_capacitance_[i];
    float calcium = old_calcium[i];
    float threshold = threshold_[i];
    if (previous_voltage <= threshold && current_voltage > threshold) {
      calcium += calcium_spike_increment_[i];
      unsigned int word_index = Bit::word(i);
      Bit::Word mask = Bit::mask(i);
      neuron_fire_bits[word_index] |= mask;
    }
    new_calcium[i] = calcium;
    new_voltage[i] = current_voltage;
    device_neuron_voltage[i] = current_voltage;
  }
  channel_buffer->release();
  state_buffer->release();
  this->publish(blank);
  return true;
}

template<>
bool NCSSimulator<ncs::sim::DeviceType::CUDA>::
update(ncs::sim::NeuronUpdateParameters* parameters) {
  std::cout << "STUB: NCSSimulator<CUDA>::update" << std::endl;
  return true;
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
