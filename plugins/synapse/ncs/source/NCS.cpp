#include <ncs/sim/AtomicWriter.h>
#include <ncs/sim/FactoryMap.h>

#include "NCS.h"

NCSDataBuffer<ncs::sim::DeviceType::CPU>::NCSDataBuffer() {
}

bool NCSDataBuffer<ncs::sim::DeviceType::CPU>::init(size_t num_synapses) {
  return true;
}

bool set(ncs::spec::Generator*& target,
         const std::string& parameter,
         ncs::spec::ModelParameters* parameters) {
  ncs::spec::Generator* generator = parameters->getGenerator(parameter);
  target = generator;
  if (!generator) {
    std::cerr << "ncs requires " << parameter << " to be defined." <<
      std::endl;
    return false;
  }
  return true;
}

void* createInstantiator(ncs::spec::ModelParameters* parameters) {
  Instantiator* instantiator = new Instantiator();
  bool result = true;
  result &= set(instantiator->utilization,
                "utilization",
                parameters);
  result &= set(instantiator->redistribution,
                "redistribution",
                parameters);
  result &= set(instantiator->last_prefire_time,
                "last_prefire_time",
                parameters);
  result &= set(instantiator->last_postfire_time,
                "last_postfire_time",
                parameters);
  result &= set(instantiator->tau_depression,
                "tau_depression",
                parameters);
  result &= set(instantiator->tau_facilitation,
                "tau_facilitation",
                parameters);
  result &= set(instantiator->tau_ltp,
                "tau_ltp",
                parameters);
  result &= set(instantiator->tau_ltd,
                "tau_ltd",
                parameters);
  result &= set(instantiator->A_ltp_minimum,
                "A_ltp_minimum",
                parameters);
  result &= set(instantiator->A_ltd_minimum,
                "A_ltd_minimum",
                parameters);
  result &= set(instantiator->max_conductance,
                "max_conductance",
                parameters);
  result &= set(instantiator->reversal_potential,
                "reversal_potential",
                parameters);
  result &= set(instantiator->tau_postsynaptic_conductance,
                "tau_postsynaptic_conductance",
                parameters);
  result &= set(instantiator->psg_waveform_duration,
                "psg_waveform_duration",
                parameters);
  result &= set(instantiator->delay,
                "delay",
                parameters);
  if (!result) {
    std::cerr << "Failed to create ncs generator" << std::endl;
    delete instantiator;
    return nullptr;
  }
  return instantiator;
}

template<ncs::sim::DeviceType::Type MType>
ncs::sim::SynapseSimulator<MType>* createSimulator() {
  return new NCSSimulator<MType>();
}

template<>
bool NCSSimulator<ncs::sim::DeviceType::CPU>::
update(ncs::sim::SynapseUpdateParameters* parameters) {
  using ncs::sim::Bit;
  const Bit::Word* synaptic_fire = parameters->synaptic_fire;
  const float* neuron_voltage = parameters->neuron_voltage;
  float* synaptic_current = parameters->synaptic_current;
  std::mutex* write_lock = parameters->write_lock;
  float simulation_time = parameters->simulation_time;
  auto old_firings = self_subscription_->pull();
  auto new_firings = this->getBlank();
  new_firings->fire_time.clear();
  new_firings->fire_index.clear();
  new_firings->psg_max.clear();

  ncs::sim::AtomicWriter<float> current_adder;
  auto add_current = [synaptic_current, 
                      neuron_voltage,
                      &current_adder](float dt,
                                      float psg_max,
                                      float tau_postsynaptic_conductance,
                                      unsigned int neuron_id,
                                      float reversal_potential) {
    float f = dt / tau_postsynaptic_conductance;
    float g = psg_max * f * exp(1.0f - f);
    float current = g * (reversal_potential - neuron_voltage[neuron_id]);
    current_adder.write(synaptic_current + neuron_id, current);
  };

  for (size_t i = 0; i < num_synapses_; ++i) {
    unsigned int fire_word_index = Bit::word(i);
    unsigned int mask = Bit::mask(i);
    if (synaptic_fire[fire_word_index] & mask) {
      float pre_dt = simulation_time - last_prefire_time_[i];
      last_prefire_time_[i] = simulation_time;
      float base_utilization = base_utilization_[i];
      float utilization = utilization_[i];
      float tau_facilitation = tau_facilitation_[i];
      float one_minus_base = 1.0f - base_utilization;
      if (tau_facilitation != 0.0f) {
        float exponent = -pre_dt / tau_facilitation;
        utilization += utilization * one_minus_base * exp(exponent);
        utilization = std::max(0.0f, std::min(utilization, 1.0f));
        utilization_[i] = utilization;
      }
      float tau_depression = tau_depression_[i];
      float redistribution = redistribution_[i];
      if (tau_depression != 0.0f) {
        float exponent = -pre_dt / tau_depression;
        float coefficient = 1.0 - redistribution * one_minus_base;
        redistribution = 1.0 - coefficient * exp(exponent);
        redistribution = std::max(0.0f, std::min(redistribution, 1.0f));
        redistribution_[i] = redistribution;
      }
      float psg_max = max_conductance_[i] * utilization * redistribution;
      new_firings->fire_time.push_back(simulation_time);
      new_firings->fire_index.push_back(i);
      new_firings->psg_max.push_back(psg_max);
      float tau_psg = tau_postsynaptic_conductance_[i];
      unsigned int neuron_id = device_neuron_device_ids_[i];
      float reversal_potential = reversal_potential_[i];
      add_current(0.0f, psg_max, tau_psg, neuron_id, reversal_potential);

      float post_dt = simulation_time - last_postfire_time_[i];
      float tau_ltp = tau_ltp_[i];
      if (tau_ltp != 0.0f) {
        float exponent = -pre_dt / tau_ltp;
        A_ltp_[i] = A_ltp_[i] * exp(exponent) + A_ltp_minimum_[i];
      }
      float tau_ltd = tau_ltd_[i];
      if (tau_ltd != 0.0f) {
        float exponent = -post_dt / tau_ltd;
        base_utilization += A_ltd_[i] * exp(exponent);
        base_utilization_[i] = base_utilization;
      }
    }
  }

  size_t num_old_firings = old_firings->fire_time.size();
  for (size_t i = 0; i < num_old_firings; ++i) {
    float fire_time = old_firings->fire_time[i];
    unsigned int fire_index = old_firings->fire_index[i];
    float psg_waveform_duration = psg_waveform_duration_[fire_index];
    float dt = simulation_time - fire_time;
    if (dt > psg_waveform_duration) {
      continue;
    }
    float psg_max = old_firings->psg_max[i];
    float tau_psg = tau_postsynaptic_conductance_[fire_index];
    unsigned int neuron_id = device_neuron_device_ids_[fire_index];
    float reversal_potential = reversal_potential_[fire_index];
    add_current(dt, psg_max, tau_psg, neuron_id, reversal_potential);
    new_firings->fire_time.push_back(fire_time);
    new_firings->fire_index.push_back(fire_index);
    new_firings->psg_max.push_back(psg_max);
  }
  old_firings->release();
  this->publish(new_firings);

  std::unique_lock<std::mutex> lock(*write_lock);
  current_adder.commit(ncs::sim::AtomicWriter<float>::Add);
  lock.unlock();
  return true;
}


template<>
bool NCSSimulator<ncs::sim::DeviceType::CUDA>::
update(ncs::sim::SynapseUpdateParameters* parameters) {
  std::cout << "STUB: NCSSimulator<CUDA>::update()" << std::endl;
  return true;
}

extern "C" {

bool load(ncs::sim::FactoryMap<ncs::sim::SynapseSimulator>* plugin_map) {
  bool result = true;
  result &= plugin_map->registerInstantiator("ncs", createInstantiator);
  
  const auto CPU = ncs::sim::DeviceType::CPU;
  result &= plugin_map->registerCPUProducer("ncs", createSimulator<CPU>);

#ifdef NCS_CUDA
  const auto CUDA = ncs::sim::DeviceType::CUDA;
  result &= plugin_map->registerCUDAProducer("ncs", createSimulator<CUDA>);
#endif
  
  return result;
}
  
}
