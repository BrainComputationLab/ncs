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
  num_channels_ = neurons_.size();
  if (!ncs::sim::Memory<MType>::malloc(neuron_plugin_ids_, num_channels_)) {
    std::cerr << "Failed to allocate memory." << std::endl;
    return false;
  }
  unsigned int* ids = new unsigned int[num_channels_];
  for (size_t i = 0; i < num_channels_; ++i) {
    ids[i] = neurons_[i]->id.plugin;
  }
  if (!ncs::sim::mem::copy<MType, ncs::sim::DeviceType::CPU>(neuron_plugin_ids_,
                                                             ids,
                                                             num_channels_)) {
    std::cerr << "Failed to copy memory." << std::endl;
    return false;
  }
  delete [] ids;
  return init_();
}

template<ncs::sim::DeviceType::Type MType>
bool ChannelSimulator<MType>::addChannel(void* instantiator,
                                         ncs::sim::Neuron* neuron) {
  instantiators_.push_back(instantiator);
  neurons_.push_back(neuron);
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



