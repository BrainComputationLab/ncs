#include <ncs/sim/AtomicWriter.h>

#include "Channel.h"

VoltageGatedIonChannel::VoltageGatedIonChannel() {
  type = Channel::VoltageGatedIon;
}

VoltageGatedIonChannel* VoltageGatedIonChannel::
instantiate(ncs::spec::ModelParameters* parameters) {
  bool result = true;
  VoltageGatedIonChannel* i = new VoltageGatedIonChannel();
  result &= parameters->get(i->v_half, "v_half");
  result &= parameters->get(i->r, "r");
  result &= parameters->get(i->activation_slope, "activation_slope");
  result &= parameters->get(i->deactivation_slope, "deactivation_slope");
  result &= parameters->get(i->equilibrium_slope, "equilibrium_slope");
  result &= parameters->get(i->conductance, "conductance");
  result &= parameters->get(i->reversal_potential, "reversal_potential");
  result &= parameters->get(i->m_initial, "m_initial");
  if (!result) {
    std::cerr << "Failed to initialize VoltageGatedIon." << std::endl;
    delete i;
    return nullptr;
  }
  return i;
}

template<>
bool VoltageGatedIonSimulator<ncs::sim::DeviceType::CPU>::
update(ChannelUpdateParameters* parameters) {
  auto old_state_buffer = state_subscription_->pull();
  const float* neuron_voltages = parameters->voltage;
  const float* old_m = old_state_buffer->getM();
  auto new_state_buffer = this->getBlank();
  float* channel_current = parameters->current;
  float* new_m = new_state_buffer->getM();
  float dt = parameters->time_step;
  ncs::sim::AtomicWriter<float> current_adder;
  for (size_t i = 0; i < num_channels_; ++i) {
    unsigned int neuron_plugin_id = neuron_plugin_ids_[i];
    float neuron_voltage = neuron_voltages[neuron_plugin_id];
    float v_half = v_half_[i];
    float d_v = neuron_voltage - v_half;
    float beta = exp(d_v * deactivation_scale_[i]);
    float alpha = exp(d_v * activation_scale_[i]);
    float tau_m = tau_scale_factor_[i] / (alpha + beta);
    float m_infinity = 1.0f / (1.0f + exp(-d_v * equilibrium_scale_[i]));
    float dt_over_tau = dt / tau_m;
    float m = old_m[i] * (1.0f - dt_over_tau) + m_infinity * dt_over_tau;
    m = std::max(0.0f, std::min(m, 1.0f));
    new_m[i] = m;
    float current = 
      conductance_[i] * m * (reversal_potential_[i] - neuron_voltage);
    current_adder.write(channel_current + neuron_plugin_id, current);
  }
  std::unique_lock<std::mutex> lock(*(parameters->write_lock));
  current_adder.commit(ncs::sim::AtomicWriter<float>::Add);
  lock.unlock();
  old_state_buffer->release();
  this->publish(new_state_buffer);
  return true;
}


template<>
bool VoltageGatedIonSimulator<ncs::sim::DeviceType::CUDA>::
update(ChannelUpdateParameters* parameters) {
  std::cout << "STUB: VoltageGatedIonSimulator<CUDA>::update" << std::endl;
  return true;
}

