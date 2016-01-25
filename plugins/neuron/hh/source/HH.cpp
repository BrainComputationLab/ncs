#include <ncs/sim/AtomicWriter.h>
#include <ncs/sim/FactoryMap.h>
#include <ncs/sim/Memory.h>

#include "HH.h"
#ifdef NCS_CUDA
#include "HH.cuh"
#endif // NCS_CUDA

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
  result &= set(vgi->reversal_potential, 
                "reversal_potential", 
                parameters, 
                "voltage_gated");
  if (!result) {
    std::cerr << "Failed to initialize VoltageGatedInstantiator." << std::endl;
    delete vgi;
    return nullptr;
  }
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
                "hh neuron");
  result &= set(instantiator->resting_potential,
                "resting_potential", 
                parameters,
                "hh neuron");
  result &= set(instantiator->capacitance, 
                "capacitance", 
                parameters,
                "hh neuron");
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
  return new HHSimulator<MType>();
}

template<>
bool VoltageGatedChannelSimulator<ncs::sim::DeviceType::CPU>::
update(ChannelUpdateParameters* parameters) {
  auto old_state = self_subscription_->pull();
  auto new_state = this->getBlank();
  // Set all particle products to 1
  for (size_t i = 0; i < num_channels_; ++i) {
    particle_products_[i] = 1.0f;
  }
  // For each particle, update x
  auto solve = [](float a,
                  float b,
                  float c, 
                  float d, 
                  float f, 
                  float h, 
                  float v) {
    float numerator = a + b * v;
    float exponent = 0.0f;
    if (f != 0.0f) {
      exponent = (v + d) / f;
    }
    float denominator = c + h * exp(exponent);
    if (denominator == 0.0f) {
      return 0.0f;
    }
    return numerator / denominator;
  };
  const float* neuron_voltages = parameters->voltage;
  float time_step = parameters->time_step;
  float dt = time_step * 1000.0f * 0.5f;
  for (size_t p = 0; p < particle_sets_.size(); ++p) {
    auto particle_set = particle_sets_[p];
    const float* old_x = old_state->x_per_level[p];
    float* new_x = new_state->x_per_level[p];
    for (size_t i = 0; i < particle_set->size; ++i) {
      float voltage = neuron_voltages[particle_set->neuron_ids[i]];
      float alpha = solve(particle_set->alpha.a[i],
                          particle_set->alpha.b[i],
                          particle_set->alpha.c[i],
                          particle_set->alpha.d[i],
                          particle_set->alpha.f[i],
                          particle_set->alpha.h[i],
                          voltage);
      float beta = solve(particle_set->beta.a[i],
                         particle_set->beta.b[i],
                         particle_set->beta.c[i],
                         particle_set->beta.d[i],
                         particle_set->beta.f[i],
                         particle_set->beta.h[i],
                         voltage);
      float tau = 1.0f / (alpha + beta);
      float x_0 = alpha * tau;
      float x = old_x[i];
      float A = x_0 / tau;
      float B = 1.0f / tau;
      float e_minus_B_dt = exp(-B * dt);
      x = x * e_minus_B_dt + A / B * (1.0 - e_minus_B_dt);
      if (x < 0.0f) x = 0.0f;
      if (x > 1.0f) x = 1.0f;
      new_x[i] = x;
      particle_products_[particle_set->particle_indices[i]] *= pow(x, particle_set->power[i]);
    }
  }
  // Atomically add current
  float* channel_current = parameters->current;
  float* channel_reversal_current = parameters->reversal_current;
  ncs::sim::AtomicWriter<float> current_adder;
  for (size_t i = 0; i < num_channels_; ++i) {
    unsigned int neuron_plugin_id = neuron_plugin_ids_[i];
    float reversal_potential = reversal_potential_[i];
    float conductance = conductance_[i];
    float forward_current = particle_products_[i] * conductance;
    current_adder.write(channel_current + neuron_plugin_id, forward_current);
    float reversal_current = particle_products_[i] * conductance * reversal_potential;
    current_adder.write(channel_reversal_current + neuron_plugin_id, reversal_current);
  }
  std::unique_lock<std::mutex> lock(*(parameters->write_lock));
  current_adder.commit(ncs::sim::AtomicWriter<float>::Add);
  lock.unlock();
  old_state->release();
  this->publish(new_state);
  return true;
}

#ifdef NCS_CUDA
template<>
bool VoltageGatedChannelSimulator<ncs::sim::DeviceType::CUDA>::
update(ChannelUpdateParameters* parameters) {
  auto old_state = self_subscription_->pull();
  auto new_state = this->getBlank();
  cuda::resetParticleProducts(particle_products_, num_channels_);
  float time_step = parameters->time_step;
  float dt = time_step * 1000.0f * 0.5f;
  const float* neuron_voltages = parameters->voltage;
  for (size_t p = 0; p < particle_sets_.size(); ++p) {
    auto particle_set = particle_sets_[p];
    const float* old_x = old_state->x_per_level[p];
    float* new_x = new_state->x_per_level[p];
    cuda::updateParticles(particle_set->alpha.a,
                          particle_set->alpha.b,
                          particle_set->alpha.c,
                          particle_set->alpha.d,
                          particle_set->alpha.f,
                          particle_set->alpha.h,
                          particle_set->beta.a,
                          particle_set->beta.b,
                          particle_set->beta.c,
                          particle_set->beta.d,
                          particle_set->beta.f,
                          particle_set->beta.h,
                          old_x,
                          neuron_voltages,
                          particle_set->neuron_ids,
                          particle_set->power,
                          particle_set->particle_indices,
                          new_x,
                          particle_products_,
                          dt,
                          particle_set->size);
  }
  float* channel_current = parameters->current;
  float* channel_reversal_current = parameters->reversal_current;
  cuda::addCurrent(neuron_plugin_ids_,
                   conductance_,
                   particle_products_,
                   reversal_potential_,
                   channel_current,
                   channel_reversal_current,
                   this->num_channels_);
  old_state->release();
  this->publish(new_state);
  return true;
}
#endif // NCS_CUDA

template<>
bool HHSimulator<ncs::sim::DeviceType::CPU>::
update(ncs::sim::NeuronUpdateParameters* parameters) {
  using ncs::sim::Bit;
  float dt = simulation_parameters_->getTimeStep() * 1000.0f;
  const float* input_current = parameters->input_current;
  const float* clamp_voltage_values = parameters->clamp_voltage_values;
  const Bit::Word* voltage_clamp_bits = parameters->voltage_clamp_bits;
  const float* synaptic_current = parameters->synaptic_current;
  float* device_neuron_voltage = parameters->neuron_voltage;
  Bit::Word* neuron_fire_bits = parameters->neuron_fire_bits;
  unsigned int num_words = Bit::num_words(num_neurons_);
  for (unsigned int i = 0; i < num_words; ++i) {
    neuron_fire_bits[i] = 0;
  }
  float step_dt = 0.5f * dt;
  for (size_t i = 0; i < 2; ++i) {
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
    const float* channel_reversal_current = channel_buffer->getReversalCurrent();
    const float* old_voltage = state_buffer->getVoltage();
    float* new_voltage = blank->getVoltage();
    for (unsigned int i = 0; i < num_neurons_; ++i) {
      float previous_voltage = old_voltage[i];
      float external_current = input_current[i] + synaptic_current[i];
      float reversal_current = channel_reversal_current[i];
      float forward_current = channel_current[i];
      float capacitance = capacitance_[i];
      float one_over_capacitance = 1.0f / capacitance;
      float A = reversal_current + external_current * one_over_capacitance;
      float B = channel_current[i] * one_over_capacitance;
      float e_minus_B_dt = exp(-B * step_dt);
      float current_voltage = 
        previous_voltage * e_minus_B_dt + A / B * (1.0f - e_minus_B_dt);
      float threshold = threshold_[i];
      if (previous_voltage <= threshold && current_voltage > threshold) {
        unsigned int word_index = Bit::word(i);
        Bit::Word mask = Bit::mask(i);
        neuron_fire_bits[word_index] |= mask;
      }
      new_voltage[i] = current_voltage;
      device_neuron_voltage[i] = current_voltage;
    }
    channel_buffer->release();
    state_buffer->release();
    this->publish(blank);
  }
  return true;
}

#ifdef NCS_CUDA
template<>
bool HHSimulator<ncs::sim::DeviceType::CUDA>::
update(ncs::sim::NeuronUpdateParameters* parameters) {
  using ncs::sim::Bit;
  float dt = simulation_parameters_->getTimeStep() * 1000.0f;
  const float* input_current = parameters->input_current;
  const float* clamp_voltage_values = parameters->clamp_voltage_values;
  const Bit::Word* voltage_clamp_bits = parameters->voltage_clamp_bits;
  const float* synaptic_current = parameters->synaptic_current;
  float* device_neuron_voltage = parameters->neuron_voltage;
  Bit::Word* neuron_fire_bits = parameters->neuron_fire_bits;
  unsigned int num_words = Bit::num_words(num_neurons_);
  ncs::sim::Memory<ncs::sim::DeviceType::CUDA>::zero(neuron_fire_bits, 
                                                     num_words);
  float step_dt = 0.5f * dt;
  for (size_t i = 0; i < 2; ++i) {
    ncs::sim::Mailbox mailbox;
    ChannelCurrentBuffer<ncs::sim::DeviceType::CUDA>* channel_buffer = nullptr;
    channel_current_subscription_->pull(&channel_buffer, &mailbox);
    NeuronBuffer<ncs::sim::DeviceType::CUDA>* state_buffer = nullptr;
    state_subscription_->pull(&state_buffer, &mailbox);
    if (!mailbox.wait(&channel_buffer, &state_buffer)) {
      return true;
    }
    auto blank = this->getBlank();
    const float* channel_current = channel_buffer->getCurrent();
    const float* channel_reversal_current = channel_buffer->getReversalCurrent();
    const float* old_voltage = state_buffer->getVoltage();
    float* new_voltage = blank->getVoltage();
    cuda::updateHH(old_voltage,
                   input_current,
                   synaptic_current,
                   channel_reversal_current,
                   channel_current,
                   capacitance_,
                   threshold_,
                   neuron_fire_bits,
                   new_voltage,
                   device_neuron_voltage,
                   step_dt,
                   num_neurons_);
    channel_buffer->release();
    state_buffer->release();
    this->publish(blank);
  }
  return true;
}
#endif // NCS_CUDA

extern "C" {

bool load(ncs::sim::FactoryMap<ncs::sim::NeuronSimulator>* plugin_map) {
  bool result = true;
  result &= plugin_map->registerInstantiator("hh", createInstantiator);

  const auto CPU = ncs::sim::DeviceType::CPU;
  result &= 
    plugin_map->registerCPUProducer("hh", createSimulator<CPU>);

#ifdef NCS_CUDA
  const auto CUDA = ncs::sim::DeviceType::CUDA;
  result &=
    plugin_map->registerCUDAProducer("hh", createSimulator<CUDA>);
#endif // NCS_CUDA
  return result;
}
  
}
