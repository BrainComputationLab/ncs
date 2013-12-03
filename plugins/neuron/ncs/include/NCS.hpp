#include <algorithm>

#include <ncs/sim/Constants.h>
#include <ncs/sim/Memory.h>

template<ncs::sim::DeviceType::Type MType>
NCSSimulator<MType>::NCSSimulator() {
  threshold_ = nullptr;
  resting_potential_ = nullptr;
  calcium_ = nullptr;
  calcium_spike_increment_ = nullptr;
  tau_calcium_ = nullptr;
  leak_reversal_potential_ = nullptr;
  tau_membrane_ = nullptr;
  r_membrane_ = nullptr;
  channel_simulators_.push_back(new VoltageGatedIonSimulator<MType>());
  channel_simulators_.push_back(new CalciumDependentSimulator<MType>());
  channel_updater_ = new ChannelUpdater<MType>();
}

template<ncs::sim::DeviceType::Type MType>
bool NCSSimulator<MType>::addNeuron(ncs::sim::Neuron* neuron) {
  neurons_.push_back(neuron);
  return true;
}

template<ncs::sim::DeviceType::Type MType>
bool NCSSimulator<MType>::
initialize(const ncs::spec::SimulationParameters* simulation_parameters) {
  using ncs::sim::Memory;
  simulation_parameters_ = simulation_parameters;
  num_neurons_ = neurons_.size();
  std::vector<float> threshold(num_neurons_);
  std::vector<float> resting_potential(num_neurons_);
  std::vector<float> calcium(num_neurons_);
  std::vector<float> calcium_spike_increment(num_neurons_);
  std::vector<float> tau_calcium(num_neurons_);
  std::vector<float> leak_reversal_potential(num_neurons_);
  std::vector<float> tau_membrane(num_neurons_);
  std::vector<float> r_membrane(num_neurons_);
  std::vector<SpikeShape*> spike_shapes;
  std::map<SpikeShape*, unsigned int> spike_shape_index_by_spike_shape;
  std::vector<unsigned int> spike_shape_index;
  for (size_t i = 0; i < num_neurons_; ++i) {
    NCSNeuron* n = (NCSNeuron*)(neurons_[i]->instantiator);
    ncs::spec::RNG rng(neurons_[i]->seed);
    threshold[i] = n->threshold->generateDouble(&rng);
    resting_potential[i] = n->resting_potential->generateDouble(&rng);
    calcium[i] = n->calcium->generateDouble(&rng);
    calcium_spike_increment[i] = 
      n->calcium_spike_increment->generateDouble(&rng);
    tau_calcium[i] = n->tau_calcium->generateDouble(&rng);
    leak_reversal_potential[i] = 
      n->leak_reversal_potential->generateDouble(&rng);
    tau_membrane[i] = n->tau_membrane->generateDouble(&rng);
    r_membrane[i] = n->r_membrane->generateDouble(&rng);
    for (auto channel : n->channels) {
      channel_simulators_[channel->type]->addChannel(channel,
                                                     neurons_[i]->id.plugin,
                                                     rng());
    }
    SpikeShape* spike_shape = n->spike_shape;
    auto search_result = spike_shape_index_by_spike_shape.find(spike_shape);
    if (spike_shape_index_by_spike_shape.end() == search_result) {
      spike_shape_index_by_spike_shape[spike_shape] = spike_shapes.size();
      spike_shapes.push_back(spike_shape);
      search_result = spike_shape_index_by_spike_shape.find(spike_shape);
    }
    spike_shape_index.push_back(search_result->second);
  }
  std::vector<float> voltage_persistence;
  std::vector<float> dt_over_capacitance;
  std::vector<float> calcium_persistence;
  float dt = simulation_parameters_->getTimeStep();
  for (size_t i = 0; i < num_neurons_; ++i) {
    float tau = tau_membrane[i];
    if (tau != 0.0f) {
      voltage_persistence.push_back(1.0f - dt / tau);
    } else {
      voltage_persistence.push_back(1.0f);
    }
    float r = r_membrane[i];
    if (r != 0.0f) {
      float capacitance = tau / r;
      dt_over_capacitance.push_back(dt / capacitance);
    } else {
      dt_over_capacitance.push_back(0.0f);
    }
    float calcium_tau = tau_calcium[i];
    if (calcium_tau != 0.0f) {
      calcium_persistence.push_back(1.0f - dt / calcium_tau);
    } else {
      calcium_persistence.push_back(1.0f);
    }
  }

  using ncs::sim::mem::clone;
  bool result = true;
  result &= clone<MType>(threshold_, threshold);
  result &= clone<MType>(resting_potential_, resting_potential);
  result &= clone<MType>(calcium_, calcium);
  result &= clone<MType>(calcium_spike_increment_, calcium_spike_increment);
  result &= clone<MType>(tau_calcium_, tau_calcium);
  result &= clone<MType>(leak_reversal_potential_, leak_reversal_potential);
  result &= clone<MType>(tau_membrane_, tau_membrane);
  result &= clone<MType>(r_membrane_, r_membrane);
  result &= clone<MType>(voltage_persistence_, voltage_persistence);
  result &= clone<MType>(dt_over_capacitance_, dt_over_capacitance);
  result &= clone<MType>(calcium_persistence_, calcium_persistence);
  if (!result) {
    std::cerr << "Failed to transfer data to device." << std::endl;
    return false;
  }

  std::vector<float> spike_shape_data;
  std::vector<unsigned int> spike_shape_offset_by_spike_shape;
  std::vector<unsigned int> spike_shape_length_by_spike_shape;
  for (auto spike_shape : spike_shapes) {
    spike_shape_offset_by_spike_shape.push_back(spike_shape_data.size());
    spike_shape_length_by_spike_shape.push_back(spike_shape->voltages.size());
    std::vector<float> voltages = spike_shape->voltages;
    std::reverse(voltages.begin(), voltages.end());
    for (float v : voltages) {
      spike_shape_data.push_back(v);
    }
  }
  if (!clone<MType>(spike_shape_data_, spike_shape_data)) {
    std::cerr << "Failed to clone spike shape data to device." << std::endl;
    return false;
  }

  std::vector<unsigned int> spike_shape_length;
  std::vector<float*> spike_shape_pointers;
  for (auto index : spike_shape_index) {
    spike_shape_length.push_back(spike_shape_length_by_spike_shape[index]);
    float* p = spike_shape_data_ + spike_shape_offset_by_spike_shape[index];
    spike_shape_pointers.push_back(p);
  }
  result &= clone<MType>(spike_shape_, spike_shape_pointers);
  result &= clone<MType>(spike_shape_length_, spike_shape_length);
  if (!result) {
    std::cerr << "Failed to clone spike shape data per neuron to device." <<
      std::endl;
    return false;
  }

  for (auto simulator : channel_simulators_) {
    if (!simulator->initialize()) {
      std::cerr << "Failed to initialize channel simulator." << std::endl;
      return false;
    }
  }

  channel_current_subscription_ = channel_updater_->subscribe();
  if (!channel_updater_->init(channel_simulators_,
                              this,
                              simulation_parameters,
                              num_neurons_,
                              ncs::sim::Constants::num_buffers)) {
    std::cerr << "Failed to initialize ChannelUpdater." << std::endl;
    return false;
  }

  if (!channel_updater_->start()) {
    std::cerr << "Failed to start ChannelUpdater." << std::endl;
    return false;
  }

  for (size_t i = 0; i < ncs::sim::Constants::num_buffers; ++i) {
    auto blank = new NeuronBuffer<MType>();
    if (!blank->init(num_neurons_)) {
      delete blank;
      std::cerr << "Failed to initialize NeuronBuffer." << std::endl;
      return false;
    }
    addBlank(blank);
  }

  state_subscription_ = this->subscribe();
  // Publish the initial state
  auto blank = this->getBlank();
  ncs::sim::mem::copy<MType, MType>(blank->getVoltage(),
                                    resting_potential_,
                                    num_neurons_);
  ncs::sim::mem::copy<MType, MType>(blank->getCalcium(),
                                    calcium_,
                                    num_neurons_);
  std::vector<int> negative_one(num_neurons_, -1);
  const auto CPU = ncs::sim::DeviceType::CPU;
  ncs::sim::mem::copy<MType, CPU>(blank->getSpikeShapeState(),
                                  negative_one.data(),
                                  num_neurons_);
  this->publish(blank);
  return true;
}

template<ncs::sim::DeviceType::Type MType>
bool NCSSimulator<MType>::initializeVoltages(float* plugin_voltages) {
  return ncs::sim::mem::copy<MType, MType>(plugin_voltages, 
                                           resting_potential_, 
                                           num_neurons_);
}

template<ncs::sim::DeviceType::Type MType>
bool NCSSimulator<MType>::update(ncs::sim::NeuronUpdateParameters* parameters) {
  std::cerr << "NCSSimulator<MType>::update is not implemented." << std::endl;
  return true;
}

template<ncs::sim::DeviceType::Type MType>
NCSSimulator<MType>::~NCSSimulator() {
  if (state_subscription_) {
    delete state_subscription_;
  }
  if (channel_current_subscription_) {
    delete channel_current_subscription_;
  }
  auto if_deletef = [](float* p) {
    if (p) {
      ncs::sim::Memory<MType>::free(p);
    }
  };
  if_deletef(threshold_);
  if_deletef(resting_potential_);
  if_deletef(calcium_);
  if_deletef(calcium_spike_increment_);
  if_deletef(tau_calcium_);
  if_deletef(leak_reversal_potential_);
  if_deletef(tau_membrane_);
  if_deletef(r_membrane_);
  if_deletef(spike_shape_data_);
  if (spike_shape_length_) {
    ncs::sim::Memory<MType>::free(spike_shape_length_);
  }
  if(spike_shape_) {
    ncs::sim::Memory<MType>::free(spike_shape_);
  }
}

