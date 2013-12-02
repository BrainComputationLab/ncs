#include <ncs/sim/AtomicWriter.h>

#include "Channel.h"
#ifdef NCS_CUDA
#include "Channel.cuh"
#endif // NCS_CUDA

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

CalciumDependentChannel::CalciumDependentChannel() {
  type = Channel::CalciumDependent;
}

CalciumDependentChannel* CalciumDependentChannel::
instantiate(ncs::spec::ModelParameters* parameters) {
  bool result = true;
  CalciumDependentChannel* c = new CalciumDependentChannel();
  result &= parameters->get(c->m_initial, "m_initial");
  result &= parameters->get(c->reversal_potential, "reversal_potential");
  result &= parameters->get(c->conductance, "conductance");
  result &= parameters->get(c->backwards_rate, "backwards_rate");
  result &= parameters->get(c->forward_scale, "forward_scale");
  result &= parameters->get(c->forward_exponent, "forward_exponent");
  result &= parameters->get(c->tau_scale, "tau_scale");
  result &= parameters->get(c->m_power, "m_power");
  if (!result) {
    std::cerr << "Failed to initialize CalciumDependentChannel." << std::endl;
    delete c;
    return false;
  }
  return c;
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

#ifdef NCS_CUDA
template<>
bool VoltageGatedIonSimulator<ncs::sim::DeviceType::CUDA>::
update(ChannelUpdateParameters* parameters) {
  auto old_state_buffer = state_subscription_->pull();
  const float* neuron_voltages = parameters->voltage;
  const float* old_m = old_state_buffer->getM();
  auto new_state_buffer = this->getBlank();
  float* channel_current = parameters->current;
  float* new_m = new_state_buffer->getM();
  float dt = parameters->time_step;
  cuda::updateVoltageGatedIon(neuron_plugin_ids_,
                              neuron_voltages,
                              v_half_,
                              deactivation_scale_,
                              activation_scale_,
                              equilibrium_scale_,
                              tau_scale_factor_,
                              old_m,
                              reversal_potential_,
                              conductance_,
                              new_m,
                              channel_current,
                              dt,
                              num_channels_);
  old_state_buffer->release();
  this->publish(new_state_buffer);
  return true;
}
#endif // NCS_CUDA

template<>
bool CalciumDependentSimulator<ncs::sim::DeviceType::CPU>::
update(ChannelUpdateParameters* parameters) {
  auto old_state_buffer = state_subscription_->pull();
  const float* neuron_voltages = parameters->voltage;
  const float* neuron_calcium = parameters->calcium;
  const float* old_m = old_state_buffer->getM();
  auto new_state_buffer = this->getBlank();
  float* channel_current = parameters->current;
  float* new_m = new_state_buffer->getM();
  float dt = parameters->time_step;
  ncs::sim::AtomicWriter<float> current_adder;
  for (size_t i = 0; i < num_channels_; ++i) {
    unsigned int neuron_plugin_id = neuron_plugin_ids_[i];
    float neuron_voltage = neuron_voltages[neuron_plugin_id];
    float calcium = neuron_calcium[neuron_plugin_id];
    float forward_rate = 
      forward_scale_[i] * pow(calcium, forward_exponent_[i]);
    float total_rate = forward_rate + backwards_rate_[i];
    float one_over_total_rate = 1.0f / total_rate;
    float tau_m = tau_scale_[i] * one_over_total_rate;
    float m_infinity = forward_rate * one_over_total_rate;
    float dt_over_tau = dt / tau_m;
    float m = old_m[i] * (1.0f - dt_over_tau) + dt_over_tau * m_infinity;
    m = std::max(0.0f, std::min(m, 1.0f));
    new_m[i] = m;
    float current = 
      conductance_[i] * pow(m, m_power_[i]) * 
      (reversal_potential_[i] - neuron_voltage);
    current_adder.write(channel_current + neuron_plugin_id, current);
  }
  std::unique_lock<std::mutex> lock(*(parameters->write_lock));
  current_adder.commit(ncs::sim::AtomicWriter<float>::Add);
  lock.unlock();
  old_state_buffer->release();
  this->publish(new_state_buffer);
  return true;
}

#ifdef NCS_CUDA
template<>
bool CalciumDependentSimulator<ncs::sim::DeviceType::CUDA>::
update(ChannelUpdateParameters* parameters) {
  auto old_state_buffer = state_subscription_->pull();
  const float* neuron_voltages = parameters->voltage;
  const float* neuron_calcium = parameters->calcium;
  const float* old_m = old_state_buffer->getM();
  auto new_state_buffer = this->getBlank();
  float* channel_current = parameters->current;
  float* new_m = new_state_buffer->getM();
  float dt = parameters->time_step;
  cuda::updateCalciumDependent(neuron_plugin_ids_,
                               neuron_voltages,
                               neuron_calcium,
                               forward_scale_,
                               forward_exponent_,
                               backwards_rate_,
                               tau_scale_,
                               m_power_,
                               conductance_,
                               reversal_potential_,
                               old_m,
                               new_m,
                               channel_current,
                               dt,
                               num_channels_);
  old_state_buffer->release();
  this->publish(new_state_buffer);
  return true;
}
#endif // NCS_CUDA

