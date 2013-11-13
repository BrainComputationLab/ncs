template<ncs::sim::DeviceType::Type MType>
VoltageGatedIonBuffer<MType>::VoltageGatedIonBuffer() {
  m_ = nullptr;
}

template<ncs::sim::DeviceType::Type MType>
bool VoltageGatedIonBuffer<MType>::init(size_t num_channels) {
  bool result = true;
  if (num_channels > 0) {
    result &= ncs::sim::Memory<MType>::malloc(m_, num_channels);
  }
  return result;
}

template<ncs::sim::DeviceType::Type MType>
float* VoltageGatedIonBuffer<MType>::getM() {
  return m_;
}

template<ncs::sim::DeviceType::Type MType>
VoltageGatedIonBuffer<MType>::~VoltageGatedIonBuffer() {
  if (m_) {
    ncs::sim::Memory<MType>::free(m_);
  }
}

template<ncs::sim::DeviceType::Type MType>
VoltageGatedIonSimulator<MType>::VoltageGatedIonSimulator() {
  v_half_ = nullptr;
  tau_scale_factor_ = nullptr;
  activation_scale_ = nullptr;
  deactivation_scale_ = nullptr;
  equilibrium_scale_ = nullptr;
  conductance_ = nullptr;
  reversal_potential_ = nullptr;
  state_subscription_ = nullptr;
}

template<ncs::sim::DeviceType::Type MType>
VoltageGatedIonSimulator<MType>::~VoltageGatedIonSimulator() {
  if (state_subscription_) {
    delete state_subscription_;
  }
  auto if_delete = [](float* p) {
    if (p) {
      ncs::sim::Memory<MType>::free(p);
    }
  };
  if_delete(v_half_);
  if_delete(tau_scale_factor_);
  if_delete(activation_scale_);
  if_delete(deactivation_scale_);
  if_delete(equilibrium_scale_);
  if_delete(conductance_);
  if_delete(reversal_potential_);
}

template<ncs::sim::DeviceType::Type MType>
bool VoltageGatedIonSimulator<MType>::init_() {
  size_t num_channels = this->num_channels_;
  std::vector<float> v_half(num_channels);
  std::vector<float> tau_scale_factor(num_channels);
  std::vector<float> activation_scale(num_channels);
  std::vector<float> deactivation_scale(num_channels);
  std::vector<float> equilibrium_scale(num_channels);
  std::vector<float> conductance(num_channels);
  std::vector<float> reversal_potential(num_channels);
  std::vector<float> m_initial(num_channels);
  for (size_t i = 0; i < num_channels; ++i) {
    auto vgi = (VoltageGatedIonChannel*)(this->instantiators_[i]);
    ncs::spec::RNG rng(this->seeds_[i]);
    v_half[i] = vgi->v_half->generateDouble(&rng);
    tau_scale_factor[i] = 1.0f / vgi->r->generateDouble(&rng);
    activation_scale[i] = 1.0f / vgi->activation_slope->generateDouble(&rng);
    deactivation_scale[i] = 
      1.0f / vgi->deactivation_slope->generateDouble(&rng);
    equilibrium_scale[i] = 1.0f / vgi->equilibrium_slope->generateDouble(&rng);
    conductance[i] = vgi->conductance->generateDouble(&rng);
    reversal_potential[i] = vgi->reversal_potential->generateDouble(&rng);
    m_initial[i] = vgi->m_initial->generateDouble(&rng);
  }
  using ncs::sim::mem::clone;
  bool result = true;
  result &= clone<MType>(v_half_, v_half);
  result &= clone<MType>(tau_scale_factor_, tau_scale_factor);
  result &= clone<MType>(activation_scale_, activation_scale);
  result &= clone<MType>(deactivation_scale_, deactivation_scale);
  result &= clone<MType>(equilibrium_scale_, equilibrium_scale);
  result &= clone<MType>(conductance_, conductance);
  result &= clone<MType>(reversal_potential_, reversal_potential);
  if (!result) {
    std::cerr << "Failed to transfer data to device." << std::endl;
    return false;
  }
  state_subscription_ = this->subscribe();
  for (size_t i = 0; i < ncs::sim::Constants::num_buffers; ++i) {
    auto blank = new VoltageGatedIonBuffer<MType>();
    if (!blank->init(num_channels)) {
      std::cerr << "Failed to initialize VoltageGatedIonBuffer." << std::endl;
      delete blank;
      return false;
    }
    addBlank(blank);
  }
  auto blank = this->getBlank();
  const auto CPU = ncs::sim::DeviceType::CPU;
  if (!ncs::sim::mem::copy<MType, CPU>(blank->getM(), 
                                       m_initial.data(), 
                                       num_channels)) {
    std::cerr << "Failed to copy initial m values." << std::endl;
    this->publish(blank);
    return false;
  }
  this->publish(blank);
  return true;
}
