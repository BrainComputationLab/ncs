#include <ncs/sim/Constants.h>
#include <ncs/sim/Memory.h>

template<ncs::sim::DeviceType::Type MType>
VoltageGatedChannelSimulator<MType>::VoltageGatedChannelSimulator() {
  particle_products_ = nullptr;
  self_subscription_ = nullptr;
}

template<ncs::sim::DeviceType::Type MType>
VoltageGatedChannelSimulator<MType>::~VoltageGatedChannelSimulator() {
  if (self_subscription_) {
    delete self_subscription_;
  }
  for (auto particle_set : particle_sets_) {
    delete particle_set;
  }
  if (particle_products_) {
    ncs::sim::Memory<MType>::free(particle_products_);
  }
}

template<ncs::sim::DeviceType::Type MType>
bool VoltageGatedChannelSimulator<MType>::
update(ChannelUpdateParameters* parameters) {
  std::cout << "STUB: VoltageGatedChannelSimulator<MType>::update()" <<
    std::endl;
  return true;
}

template<ncs::sim::DeviceType::Type MType>
bool VoltageGatedChannelSimulator<MType>::init_() {
  if (this->instantiators_.empty()) {
    return true;
  }

  size_t max_particles_per_channel = 0;
  for (auto i : this->instantiators_) {
    auto instantiator = (VoltageGatedInstantiator*)i;
    max_particles_per_channel = std::max(max_particles_per_channel,
                                         instantiator->particles.size());
  }

  std::vector<size_t> num_particles_per_level(max_particles_per_channel, 0);
  for (auto i : this->instantiators_) {
    auto instantiator = (VoltageGatedInstantiator*)i;
    num_particles_per_level[instantiator->particles.size() - 1]++;
  }
  for (size_t i = max_particles_per_channel - 1; i > 0; --i) {
    num_particles_per_level[i - 1] += num_particles_per_level[i];
  }

  bool result = true;
  const auto CPU = ncs::sim::DeviceType::CPU;
  std::vector<ParticleSet<CPU>> particle_sets(max_particles_per_channel);
  particle_sets_.resize(max_particles_per_channel);
  for (size_t i = 0; i < max_particles_per_channel; ++i) {
    if (!particle_sets[i].init(num_particles_per_level[i])) {
      std::cerr << "Failed to allocate CPU particle buffer." << std::endl;
      result = false;
    }
    particle_sets_[i] = new ParticleSet<MType>();
    if (!particle_sets_[i]->init(num_particles_per_level[i])) {
      std::cerr << "Failed to allocate MType particle buffer." << std::endl;
      result = false;
    }
  }
  if (!result) {
    return false;
  }

  auto generate_particle = [=](ParticleConstants<CPU>& p,
                               void* instantiator,
                               ncs::spec::RNG* rng,
                               size_t index) {
    auto pci = (ParticleConstantsInstantiator*)instantiator;
    p.a[index] = pci->a->generateDouble(rng);
    p.b[index] = pci->b->generateDouble(rng);
    p.c[index] = pci->c->generateDouble(rng);
    p.d[index] = pci->d->generateDouble(rng);
    p.f[index] = pci->f->generateDouble(rng);
    p.h[index] = pci->h->generateDouble(rng);
  };
  size_t num_channels = this->num_channels_;
  std::vector<float> conductance(num_channels);
  std::vector<float> reversal_potential(num_channels);
  std::vector<size_t> particles_used_per_level(max_particles_per_channel, 0);
  for (size_t i = 0; i < num_channels; ++i) {
    auto ci = (VoltageGatedInstantiator*)(this->instantiators_[i]);
    auto seed = this->seeds_[i];
    ncs::spec::RNG rng(seed);
    conductance[i] = ci->conductance->generateDouble(&rng);
    reversal_potential[i] = ci->reversal_potential->generateDouble(&rng);
    for (size_t j = 0; j < ci->particles.size(); ++j) {
      auto vpi = (VoltageParticleInstantiator*)(ci->particles[j]);
      size_t k = particles_used_per_level[j];
      particle_sets[j].power[k] = vpi->power->generateDouble(&rng);
      particle_sets[j].x_initial[k] = vpi->x_initial->generateDouble(&rng);
      generate_particle(particle_sets[j].alpha, vpi->alpha, &rng, k);
      generate_particle(particle_sets[j].beta, vpi->beta, &rng, k);
      particle_sets[j].particle_indices[k] = i;
      particle_sets[j].neuron_ids[k] = this->cpu_neuron_plugin_ids_[i];
      particles_used_per_level[j]++;
    }
  }
  result &= ncs::sim::Memory<MType>::malloc(particle_products_, 
                                            this->num_channels_);
  using ncs::sim::mem::clone;
  result &= clone<MType>(conductance_, conductance);
  result &= clone<MType>(reversal_potential_, reversal_potential);
  for (size_t i = 0; i < max_particles_per_channel; ++i) {
    result &= particle_sets_[i]->copyFrom(particle_sets.data() + i);
  }
  if (!result) {
    std::cerr << "Failed to transfer data to device." << std::endl;
    return false;
  }

  for (size_t i = 0; i < ncs::sim::Constants::num_buffers; ++i) {
    auto blank = new VoltageGatedBuffer<MType>();
    if (!blank->init(num_particles_per_level)) {
      std::cerr << "Failed to initialize VoltageGatedBuffer." << std::endl;
      delete blank;
      return false;
    }
    this->addBlank(blank);
  }
  self_subscription_ = this->subscribe();
  auto blank = this->getBlank();
  using ncs::sim::mem::copy;
  for (size_t i = 0; i < particle_sets_.size(); ++i) {
    result &= copy<MType, MType>(blank->x_per_level[i],
                                 particle_sets_[i]->x_initial, 
                                 particle_sets_[i]->size);
  }
  this->publish(blank);
  if (!result) {
    std::cerr << "Failed to transfer initial particle data." << std::endl;
    return false;
  }

  return true;
}

template<ncs::sim::DeviceType::Type MType>
ParticleConstants<MType>::ParticleConstants()
  : a(nullptr),
    b(nullptr),
    c(nullptr),
    d(nullptr),
    f(nullptr),
    h(nullptr) {
}

template<ncs::sim::DeviceType::Type MType>
bool ParticleConstants<MType>::init(size_t num_constants) {
  bool result = true;
  result &= ncs::sim::Memory<MType>::malloc(a, num_constants);
  result &= ncs::sim::Memory<MType>::malloc(b, num_constants);
  result &= ncs::sim::Memory<MType>::malloc(c, num_constants);
  result &= ncs::sim::Memory<MType>::malloc(d, num_constants);
  result &= ncs::sim::Memory<MType>::malloc(f, num_constants);
  result &= ncs::sim::Memory<MType>::malloc(h, num_constants);
  return result;
}

template<ncs::sim::DeviceType::Type MType>
template<ncs::sim::DeviceType::Type SType>
bool ParticleConstants<MType>::copyFrom(ParticleConstants<SType>* source,
                                        size_t num_constants) {
  bool result = true;
  result &= ncs::sim::mem::copy<MType, SType>(a, source->a, num_constants);
  result &= ncs::sim::mem::copy<MType, SType>(b, source->b, num_constants);
  result &= ncs::sim::mem::copy<MType, SType>(c, source->c, num_constants);
  result &= ncs::sim::mem::copy<MType, SType>(d, source->d, num_constants);
  result &= ncs::sim::mem::copy<MType, SType>(f, source->f, num_constants);
  result &= ncs::sim::mem::copy<MType, SType>(h, source->h, num_constants);
  return result;
}

template<ncs::sim::DeviceType::Type MType>
ParticleConstants<MType>::~ParticleConstants() {
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
ParticleSet<MType>::ParticleSet() {
  particle_indices = nullptr;
  power = nullptr;
  x_initial = nullptr;
  neuron_ids = nullptr;
  size = 0;
}

template<ncs::sim::DeviceType::Type MType>
bool ParticleSet<MType>::init(size_t count) {
  size = count;
  bool result = true;
  result &= alpha.init(count);
  result &= beta.init(count);
  result &= ncs::sim::Memory<MType>::malloc(particle_indices, size);
  result &= ncs::sim::Memory<MType>::malloc(power, size);
  result &= ncs::sim::Memory<MType>::malloc(x_initial, size);
  result &= ncs::sim::Memory<MType>::malloc(neuron_ids, size);
  return result;
}

template<ncs::sim::DeviceType::Type MType>
template<ncs::sim::DeviceType::Type SType>
bool ParticleSet<MType>::copyFrom(ParticleSet<SType>* s) {
  using ncs::sim::mem::copy;
  bool result = true;
  result &= alpha.copyFrom(&(s->alpha), size);
  result &= beta.copyFrom(&(s->beta), size);
  result &= copy<MType, SType>(particle_indices, s->particle_indices, size);
  result &= copy<MType, SType>(neuron_ids, s->neuron_ids, size);
  result &= copy<MType, SType>(power, s->power, size);
  result &= copy<MType, SType>(x_initial, s->x_initial, size);
  return result;
}

template<ncs::sim::DeviceType::Type MType>
ParticleSet<MType>::~ParticleSet() {
  if (particle_indices) {
    ncs::sim::Memory<MType>::free(particle_indices);
  }
  if (power) {
    ncs::sim::Memory<MType>::free(power);
  }
  if (x_initial) {
    ncs::sim::Memory<MType>::free(x_initial);
  }
  if (neuron_ids) {
    ncs::sim::Memory<MType>::free(neuron_ids);
  }
}

template<ncs::sim::DeviceType::Type MType>
VoltageGatedBuffer<MType>::VoltageGatedBuffer() {
}

template<ncs::sim::DeviceType::Type MType>
bool VoltageGatedBuffer<MType>::init(const std::vector<size_t>& counts_per_level) {
  size_per_level = counts_per_level;
  bool result = true;
  for (size_t i = 0; i < size_per_level.size(); ++i) {
    size_t size = size_per_level[i];
    float* x = nullptr;
    result &= ncs::sim::Memory<MType>::malloc(x, size);
    if (result) {
      x_per_level.push_back(x);
    }
  }
  return result;
}

template<ncs::sim::DeviceType::Type MType>
VoltageGatedBuffer<MType>::~VoltageGatedBuffer() {
  for (auto x : x_per_level) {
    ncs::sim::Memory<MType>::free(x);
  }
}

template<ncs::sim::DeviceType::Type MType>
HHSimulator<MType>::HHSimulator() {
  channel_simulators_.push_back(new VoltageGatedChannelSimulator<MType>());
  threshold_ = nullptr;
  resting_potential_ = nullptr;
  capacitance_ = nullptr;
  channel_current_subscription_ = nullptr;
  channel_updater_ = new ChannelUpdater<MType>();
}

template<ncs::sim::DeviceType::Type MType>
bool HHSimulator<MType>::addNeuron(ncs::sim::Neuron* neuron) {
  neurons_.push_back(neuron);
  return true;
}

template<ncs::sim::DeviceType::Type MType>
bool HHSimulator<MType>::
initialize(const ncs::spec::SimulationParameters* simulation_parameters) {
  simulation_parameters_ = simulation_parameters;
  using ncs::sim::Memory;
  num_neurons_ = neurons_.size();
  bool result = true;
  result &= Memory<MType>::malloc(threshold_, num_neurons_);
  result &= Memory<MType>::malloc(resting_potential_, num_neurons_);
  result &= Memory<MType>::malloc(capacitance_, num_neurons_);
  if (!result) {
    std::cerr << "Failed to allocate memory." << std::endl;
    return false;
  }
  
  float* threshold = new float[num_neurons_];
  float* resting_potential = new float[num_neurons_];
  float* capacitance = new float[num_neurons_];
  for (size_t i = 0; i < num_neurons_; ++i) {
    NeuronInstantiator* ni = (NeuronInstantiator*)(neurons_[i]->instantiator);
    ncs::spec::RNG rng(neurons_[i]->seed);
    threshold[i] = ni->threshold->generateDouble(&rng);
    resting_potential[i] = ni->resting_potential->generateDouble(&rng);
    capacitance[i] = ni->capacitance->generateDouble(&rng);
    unsigned int plugin_index = neurons_[i]->id.plugin;
    for (auto channel : ni->channels) {
      auto channel_type = channel->type;
      channel_simulators_[channel_type]->addChannel(channel, plugin_index, rng());
    }
  }

  const auto CPU = ncs::sim::DeviceType::CPU;
  auto copy = [this](float* src, float* dst) {
    return ncs::sim::mem::copy<CPU, MType>(src, dst, num_neurons_);
  };
  result &= copy(threshold, threshold_);
  result &= copy(resting_potential, resting_potential_);
  result &= copy(capacitance, capacitance_);

  delete [] threshold;
  delete [] resting_potential;
  delete [] capacitance;

  if (!result) {
    std::cerr << "Failed to copy data." << std::endl;
    return false;
  }

  for (size_t i = 0; i < ncs::sim::Constants::num_buffers; ++i) {
    auto blank = new NeuronBuffer<MType>();
    if (!blank->init(num_neurons_)) {
      delete blank;
      std::cerr << "Failed to initialize NeuronBuffer." << std::endl;
      return false;
    }
    this->addBlank(blank);
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

  state_subscription_ = this->subscribe();

  // Publish the initial state
  auto blank = this->getBlank();
  ncs::sim::mem::copy<MType, MType>(blank->getVoltage(),
                                    resting_potential_,
                                    num_neurons_);
  this->publish(blank);
  return true;
}

template<ncs::sim::DeviceType::Type MType>
bool HHSimulator<MType>::initializeVoltages(float* plugin_voltages) {
  return ncs::sim::mem::copy<MType, MType>(plugin_voltages, 
                                           resting_potential_, 
                                           num_neurons_);
}

template<ncs::sim::DeviceType::Type MType>
bool HHSimulator<MType>::
update(ncs::sim::NeuronUpdateParameters* parameters) {
  std::cout << "STUB: HHSimulator<MType>::update()" << std::endl;
  return true;
}

template<ncs::sim::DeviceType::Type MType>
HHSimulator<MType>::~HHSimulator() {
  if (state_subscription_) {
    delete state_subscription_;
  }
  if (channel_current_subscription_) {
    delete channel_current_subscription_;
  }
  auto if_delete = [](float* p) {
    if (p) {
      ncs::sim::Memory<MType>::free(p);
    }
  };
  if_delete(threshold_);
  if_delete(resting_potential_);
  if_delete(capacitance_);
}
