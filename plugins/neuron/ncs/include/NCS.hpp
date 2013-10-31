#include <ncs/sim/Constants.h>

template<ncs::sim::DeviceType::Type MType>
ChannelCurrentBuffer<MType>::ChannelCurrentBuffer() 
  : current_(nullptr) {
}

template<ncs::sim::DeviceType::Type MType>
bool ChannelCurrentBuffer<MType>::init(size_t num_neurons) {
  if (num_neurons > 0) {
    return ncs::sim::Memory<MType>::malloc(current_, num_neurons);
  }
  return true;
}

template<ncs::sim::DeviceType::Type MType>
float* ChannelCurrentBuffer<MType>::getCurrent() {
  return current_;
}

template<ncs::sim::DeviceType::Type MType>
ChannelCurrentBuffer<MType>::~ChannelCurrentBuffer() {
  if (current_) {
    ncs::sim::Memory<MType>::free(current_);
  }
}

template<ncs::sim::DeviceType::Type MType>
ChannelSimulator<MType>::ChannelSimulator() 
  : neuron_plugin_ids_(nullptr) {
}

template<ncs::sim::DeviceType::Type MType>
bool ChannelSimulator<MType>::initialize() {
  num_channels_ = cpu_neuron_plugin_ids_.size();
  if (!ncs::sim::Memory<MType>::malloc(neuron_plugin_ids_, num_channels_)) {
    std::cerr << "Failed to allocate memory." << std::endl;
    return false;
  }
  const auto CPU = ncs::sim::DeviceType::CPU;
  using namespace ncs::sim;
  if (!ncs::sim::mem::copy<MType, CPU>(neuron_plugin_ids_,
                                       cpu_neuron_plugin_ids_.data(),
                                       num_channels_)) {
    std::cerr << "Failed to copy memory." << std::endl;
    return false;
  }
  return init_();
}

template<ncs::sim::DeviceType::Type MType>
bool ChannelSimulator<MType>::addChannel(void* instantiator,
                                         unsigned int neuron_plugin_id,
                                         int seed) {
  instantiators_.push_back(instantiator);
  cpu_neuron_plugin_ids_.push_back(neuron_plugin_id);
  seeds_.push_back(seed);
  return true;
}

template<ncs::sim::DeviceType::Type MType>
ChannelSimulator<MType>::~ChannelSimulator() {
  if (neuron_plugin_ids_) {
    ncs::sim::Memory<MType>::free(neuron_plugin_ids_);
  }
}

template<ncs::sim::DeviceType::Type MType>
VoltageGatedChannelSimulator<MType>::VoltageGatedChannelSimulator() {
}

template<ncs::sim::DeviceType::Type MType>
VoltageGatedChannelSimulator<MType>::~VoltageGatedChannelSimulator() {
}

template<ncs::sim::DeviceType::Type MType>
bool VoltageGatedChannelSimulator<MType>::init_() {
}

template<ncs::sim::DeviceType::Type MType>
VoltageGatedChannelSimulator<MType>::ParticleConstants::ParticleConstants()
  : a(nullptr),
    b(nullptr),
    c(nullptr),
    d(nullptr),
    f(nullptr),
    h(nullptr) {
}

template<ncs::sim::DeviceType::Type MType>
VoltageGatedChannelSimulator<MType>::ParticleConstants::~ParticleConstants() {
  if (a) {
    ncs::sim::Memory<MType>::free(a);
  }
  if (b) {
    ncs::sim::Memory<MType>::free(b);
  }
  if (c) {
    ncs::sim::Memory<MType>::free(c);
  }
  if (d) {
    ncs::sim::Memory<MType>::free(d);
  }
  if (f) {
    ncs::sim::Memory<MType>::free(f);
  }
  if (h) {
    ncs::sim::Memory<MType>::free(h);
  }
}

template<ncs::sim::DeviceType::Type MType>
NCSSimulator<MType>::NCSSimulator() {
  // TODO(rvhoang): add calcium simulator here
  channel_simulators_.push_back(new VoltageGatedChannelSimulator<MType>());
  threshold_ = nullptr;
  resting_potential_ = nullptr;
  calcium_ = nullptr;
  calcium_spike_increment_ = nullptr;
  tau_calcium_ = nullptr;
  leak_reversal_potential_ = nullptr;
  leak_conductance_ = nullptr;
  tau_membrane_ = nullptr;
  r_membrane_ = nullptr;
  channel_current_subscription_ = nullptr;
}

template<ncs::sim::DeviceType::Type MType>
bool NCSSimulator<MType>::addNeuron(ncs::sim::Neuron* neuron) {
  neurons_.push_back(neuron);
  return true;
}

template<ncs::sim::DeviceType::Type MType>
bool NCSSimulator<MType>::initialize() {
  using ncs::sim::Memory;
  num_neurons_ = neurons_.size();
  bool result = true;
  result &= Memory<MType>::malloc(threshold_, num_neurons_);
  result &= Memory<MType>::malloc(resting_potential_, num_neurons_);
  result &= Memory<MType>::malloc(calcium_, num_neurons_);
  result &= Memory<MType>::malloc(calcium_spike_increment_, num_neurons_);
  result &= Memory<MType>::malloc(tau_calcium_, num_neurons_);
  result &= Memory<MType>::malloc(leak_reversal_potential_, num_neurons_);
  result &= Memory<MType>::malloc(leak_conductance_, num_neurons_);
  result &= Memory<MType>::malloc(tau_membrane_, num_neurons_);
  result &= Memory<MType>::malloc(r_membrane_, num_neurons_);
  if (!result) {
    std::cerr << "Failed to allocate memory." << std::endl;
    return false;
  }
  
  float* threshold = new float[num_neurons_];
  float* resting_potential = new float[num_neurons_];
  float* calcium = new float[num_neurons_];
  float* calcium_spike_increment = new float[num_neurons_];
  float* tau_calcium = new float[num_neurons_];
  float* leak_reversal_potential = new float[num_neurons_];
  float* leak_conductance = new float[num_neurons_];
  float* tau_membrane = new float[num_neurons_];
  float* r_membrane = new float[num_neurons_];
  for (size_t i = 0; i < num_neurons_; ++i) {
    NeuronInstantiator* ni = (NeuronInstantiator*)(neurons_[i]->instantiator);
    ncs::spec::RNG rng(neurons_[i]->seed);
    threshold[i] = ni->threshold->generateDouble(&rng);
    resting_potential[i] = ni->resting_potential->generateDouble(&rng);
    calcium[i] = ni->calcium->generateDouble(&rng);
    calcium_spike_increment[i] = ni->calcium_spike_increment->generateDouble(&rng);
    tau_calcium[i] = ni->tau_calcium->generateDouble(&rng);
    leak_reversal_potential[i] = ni->leak_reversal_potential->generateDouble(&rng);
    leak_conductance[i] = ni->leak_conductance->generateDouble(&rng);
    tau_membrane[i] = ni->tau_membrane->generateDouble(&rng);
    r_membrane[i] = ni->r_membrane->generateDouble(&rng);
    unsigned int plugin_index = neurons_[i]->id.plugin;
    for (auto channel : ni->channels) {
      auto channel_type = channel->type;
      channel_simulators_[channel_type]->addChannel(channel, plugin_index, rng());
    }
  }

  const auto CPU = ncs::sim::DeviceType::CPU;
  auto copy = [num_neurons_](float* src, float* dst) {
    return Memory<CPU>::To<MType>::copy(src, dst, num_neurons_);
  };
  result &= copy(threshold, threshold_);
  result &= copy(resting_potential, resting_potential_);
  result &= copy(calcium, calcium_);
  result &= copy(calcium_spike_increment, calcium_spike_increment_);
  result &= copy(tau_calcium, tau_calcium_);
  result &= copy(leak_reversal_potential, leak_reversal_potential_);
  result &= copy(leak_conductance, leak_conductance_);
  result &= copy(tau_membrane, tau_membrane_);
  result &= copy(r_membrane, r_membrane_);

  delete [] threshold;
  delete [] resting_potential;
  delete [] calcium;
  delete [] calcium_spike_increment;
  delete [] tau_calcium;
  delete [] leak_reversal_potential;
  delete [] leak_conductance;
  delete [] tau_membrane;
  delete [] r_membrane;

  if (!result) {
    std::cerr << "Failed to copy data." << std::endl;
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
                              num_neurons_,
                              ncs::sim::Constants::num_buffers)) {
    std::cerr << "Failed to initialize ChannelUpdater." << std::endl;
    return false;
  }

  return true;
}

template<ncs::sim::DeviceType::Type MType>
bool NCSSimulator<MType>::initializeVoltages(float* plugin_voltages) {
}

template<ncs::sim::DeviceType::Type MType>
NCSSimulator<MType>::~NCSSimulator() {
  auto if_delete = [](float* p) {
    if (p) {
      ncs::sim::Memory<MType>::free(p);
    }
  };
  if_delete(threshold_);
  if_delete(resting_potential_);
  if_delete(calcium_);
  if_delete(calcium_spike_increment_);
  if_delete(tau_calcium_);
  if_delete(leak_reversal_potential_);
  if_delete(leak_conductance_);
  if_delete(tau_membrane_);
  if_delete(r_membrane_);
}
