#include <ncs/sim/FactoryMap.h>

#include "NCS.h"
#ifdef NCS_CUDA
#include "NCS.cuh"
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

NCSNeuron::NCSNeuron() {
  spike_shape = nullptr;
}

NCSNeuron::~NCSNeuron() {
  if (spike_shape) {
    delete spike_shape;
  }
}

void* NCSNeuron::instantiate(ncs::spec::ModelParameters* parameters) {
  NCSNeuron* n = new NCSNeuron;
  bool result = true;
  result &= set(n->threshold, "threshold", parameters, "ncs_neuron");
  result &= set(n->resting_potential,
                "resting_potential",
                parameters,
                "ncs_neuron");
  result &= set(n->calcium, "calcium", parameters, "ncs_neuron");
  result &= set(n->calcium_spike_increment,
                "calcium_spike_increment",
                parameters,
                "ncs_neuron");
  result &= set(n->tau_calcium, "tau_calcium", parameters, "ncs_neuron");
  result &= set(n->leak_reversal_potential,
                "leak_reversal_potential",
                parameters,
                "ncs_neuron");
  result &= set(n->tau_membrane, "tau_membrane", parameters, "ncs_neuron");
  result &= set(n->r_membrane, "r_membrane", parameters, "ncs_neuron");
  auto spike_shape_generator = parameters->getGenerator("spike_shape");
  if (nullptr == spike_shape_generator) {
    std::cerr << "ncs_neuron requires spike_shape to be defined." << std::endl;
    delete n;
    return nullptr;
  }
  ncs::spec::RNG rng(0);
  auto spike_shape_values = spike_shape_generator->generateList(&rng);
  if (spike_shape_values.empty()) {
    std::cerr << "ncs_neuron spike_shape must be nonempty." << std::endl;
    delete n;
    return nullptr;
  }
  SpikeShape* spike_shape = new SpikeShape();
  for (auto gen : spike_shape_values) {
    spike_shape->voltages.push_back(gen->generateDouble(&rng));
  }
  n->spike_shape = spike_shape;
  if (!result) {
    std::cerr << "Failed to initialize NCSNeuron." << std::endl;
    delete n;
    return nullptr;
  }

  auto channel_list_generator = parameters->getGenerator("channels");
  if (channel_list_generator) {
    auto channel_list = channel_list_generator->generateList(&rng);
    Channel* channel = nullptr;
    for (auto channel_generator : channel_list) {
      auto channel_spec = channel_generator->generateParameters(&rng);
      auto type = channel_spec->getType();
      if (type == "voltage_gated_ion") {
        channel = VoltageGatedIonChannel::instantiate(channel_spec);
      } else if (type == "calcium_dependent") {
        channel = CalciumDependentChannel::instantiate(channel_spec);
      } else {
        std::cerr << "Unrecognized Channel type " << type << std::endl;
      }
      if (nullptr == channel) {
        std::cerr << "Failed to create channel." << std::endl;
        delete n;
        return nullptr;
      }
      n->channels.push_back(channel);
    }
  }
  return n;
}

template<>
bool NCSSimulator<ncs::sim::DeviceType::CPU>::
update(ncs::sim::NeuronUpdateParameters* parameters) {
  using ncs::sim::Bit;
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

  ncs::sim::Mailbox mailbox;
  ChannelCurrentBuffer<ncs::sim::DeviceType::CPU>* channel_buffer = nullptr;
  channel_current_subscription_->pull(&channel_buffer, &mailbox);
  NeuronBuffer<ncs::sim::DeviceType::CPU>* state_buffer = nullptr;
  state_subscription_->pull(&state_buffer, &mailbox);
  if (!mailbox.wait(&channel_buffer, &state_buffer)) {
    state_subscription_->cancel();
    channel_current_subscription_->cancel();
    if (state_buffer) {
      state_buffer->release();
    }
    if (channel_buffer) {
      channel_buffer->release();
    }
    return true;
  }
  const float* old_voltage = state_buffer->getVoltage();
  const float* old_calcium = state_buffer->getCalcium();
  const int* old_spike_shape_state = state_buffer->getSpikeShapeState();
  const float* channel_current = channel_buffer->getCurrent();
  auto blank = this->getBlank();
  float* new_voltage = blank->getVoltage();
  float* new_calcium = blank->getCalcium();
  int* new_spike_shape_state = blank->getSpikeShapeState();
  for (size_t i = 0; i < num_neurons_; ++i) {
    int spike_shape_state = old_spike_shape_state[i];
    float voltage = old_voltage[i];
    float calcium = old_calcium[i];
    float total_current = input_current[i] + 
                          synaptic_current[i] + channel_current[i];
    if (spike_shape_state < 0) { // Do real computations
      float resting_potential = resting_potential_[i];
      float dv = voltage - resting_potential;
      voltage = resting_potential + 
                dv * voltage_persistence_[i] + 
                dt_over_capacitance_[i] * total_current;
      if (voltage > threshold_[i]) {
        spike_shape_state = spike_shape_length_[i] - 1;
        calcium += calcium_spike_increment_[i];
        unsigned int word_index = Bit::word(i);
        Bit::Word mask = Bit::mask(i);
        neuron_fire_bits[word_index] |= mask;
      }
    }
    if (spike_shape_state >= 0) { // Still following spike shape
      voltage = spike_shape_[i][spike_shape_state];
      spike_shape_state--;
    }
    new_voltage[i] = voltage;
    new_spike_shape_state[i] = spike_shape_state;
    new_calcium[i] = calcium;
    device_neuron_voltage[i] = voltage;
  }
  state_buffer->release();
  channel_buffer->release();
  this->publish(blank);
  return true;
}

#ifdef NCS_CUDA
template<>
bool NCSSimulator<ncs::sim::DeviceType::CUDA>::
update(ncs::sim::NeuronUpdateParameters* parameters) {
  using ncs::sim::Bit;
  const float* input_current = parameters->input_current;
  const float* clamp_voltage_values = parameters->clamp_voltage_values;
  const Bit::Word* voltage_clamp_bits = parameters->voltage_clamp_bits;
  const float* synaptic_current = parameters->synaptic_current;
  float* device_neuron_voltage = parameters->neuron_voltage;
  Bit::Word* neuron_fire_bits = parameters->neuron_fire_bits;

  ncs::sim::Mailbox mailbox;
  ChannelCurrentBuffer<ncs::sim::DeviceType::CUDA>* channel_buffer = nullptr;
  channel_current_subscription_->pull(&channel_buffer, &mailbox);
  NeuronBuffer<ncs::sim::DeviceType::CUDA>* state_buffer = nullptr;
  state_subscription_->pull(&state_buffer, &mailbox);
  if (!mailbox.wait(&channel_buffer, &state_buffer)) {
    state_subscription_->cancel();
    channel_current_subscription_->cancel();
    if (state_buffer) {
      state_buffer->release();
    }
    if (channel_buffer) {
      channel_buffer->release();
    }
    return true;
  }
  const float* old_voltage = state_buffer->getVoltage();
  const float* old_calcium = state_buffer->getCalcium();
  const int* old_spike_shape_state = state_buffer->getSpikeShapeState();
  const float* channel_current = channel_buffer->getCurrent();
  auto blank = this->getBlank();
  float* new_voltage = blank->getVoltage();
  float* new_calcium = blank->getCalcium();
  int* new_spike_shape_state = blank->getSpikeShapeState();
  cuda::updateNeurons(old_spike_shape_state,
                      old_voltage,
                      old_calcium,
                      input_current,
                      synaptic_current,
                      channel_current,
                      resting_potential_,
                      voltage_persistence_,
                      dt_over_capacitance_,
                      spike_shape_length_,
                      calcium_spike_increment_,
                      spike_shape_,
                      threshold_,
                      neuron_fire_bits,
                      new_voltage,
                      new_spike_shape_state,
                      new_calcium,
                      device_neuron_voltage,
                      num_neurons_);

  state_buffer->release();
  channel_buffer->release();
  this->publish(blank);
  return true;
}
#endif // NCS_CUDA 

template<ncs::sim::DeviceType::Type MType>
ncs::sim::NeuronSimulator<MType>* createSimulator() {
  return new NCSSimulator<MType>();
}

extern "C" {

bool load(ncs::sim::FactoryMap<ncs::sim::NeuronSimulator>* plugin_map) {
  bool result = true;
  result &= plugin_map->registerInstantiator("ncs", NCSNeuron::instantiate);

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
