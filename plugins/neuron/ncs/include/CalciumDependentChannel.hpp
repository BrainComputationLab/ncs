template<ncs::sim::DeviceType::Type MType>
CalciumDependentBuffer<MType>::CalciumDependentBuffer() {
  m_ = nullptr;
}

template<ncs::sim::DeviceType::Type MType>
bool CalciumDependentBuffer<MType>::init(size_t num_channels) {
  bool result = true;
  if (num_channels > 0) {
    result &= ncs::sim::Memory<MType>::malloc(m_, num_channels);
  }
  return result;
}

template<ncs::sim::DeviceType::Type MType>
float* CalciumDependentBuffer<MType>::getM() {
  return m_;
}

template<ncs::sim::DeviceType::Type MType>
CalciumDependentBuffer<MType>::~CalciumDependentBuffer() {
  if (m_) {
    ncs::sim::Memory<MType>::free(m_);
  }
}

template<ncs::sim::DeviceType::Type MType>
CalciumDependentSimulator<MType>::CalciumDependentSimulator() {
  reversal_potential_ = nullptr;
  conductance_ = nullptr;
  backwards_rate_ = nullptr;
  forward_scale_ = nullptr;
  forward_exponent_ = nullptr;
  tau_scale_ = nullptr;
  state_subscription_ = nullptr;
  m_power_ = nullptr;
}

template<ncs::sim::DeviceType::Type MType>
CalciumDependentSimulator<MType>::~CalciumDependentSimulator() {
  if (state_subscription_) {
    delete state_subscription_;
  }
  auto if_delete = [](float* p) {
    if (p) {
      ncs::sim::Memory<MType>::free(p);
    }
  };
  if_delete(reversal_potential_);
  if_delete(conductance_);
  if_delete(backwards_rate_);
  if_delete(forward_scale_);
  if_delete(forward_exponent_);
  if_delete(tau_scale_);
  if_delete(m_power_);
}

template<ncs::sim::DeviceType::Type MType>
bool CalciumDependentSimulator<MType>::init_() {
  size_t num_channels = this->num_channels_;
  std::vector<float> m_initial(num_channels);
  std::vector<float> reversal_potential(num_channels);
  std::vector<float> conductance(num_channels);
  std::vector<float> backwards_rate(num_channels);
  std::vector<float> forward_scale(num_channels);
  std::vector<float> forward_exponent(num_channels);
  std::vector<float> tau_scale(num_channels);
  std::vector<float> m_power(num_channels);
  for (size_t i = 0; i < num_channels; ++i) {
    auto cdc = (CalciumDependentChannel*)(this->instantiators_[i]);
    ncs::spec::RNG rng(this->seeds_[i]);
    reversal_potential[i] = cdc->reversal_potential->generateDouble(&rng);
    conductance[i] = cdc->conductance->generateDouble(&rng);
    backwards_rate[i] = cdc->backwards_rate->generateDouble(&rng);
    forward_scale[i] = cdc->forward_scale->generateDouble(&rng);
    forward_exponent[i] = cdc->forward_exponent->generateDouble(&rng);
    tau_scale[i] = cdc->tau_scale->generateDouble(&rng);
    m_power[i] = cdc->m_power->generateDouble(&rng);
  }
  using ncs::sim::mem::clone;
  bool result = true;
  result &= clone<MType>(reversal_potential_, reversal_potential);
  result &= clone<MType>(conductance_, conductance);
  result &= clone<MType>(backwards_rate_, backwards_rate);
  result &= clone<MType>(forward_scale_, forward_scale);
  result &= clone<MType>(forward_exponent_, forward_exponent);
  result &= clone<MType>(tau_scale_, tau_scale);
  result &= clone<MType>(m_power_, m_power);
  if (!result) {
    std::cerr << "Failed to transfer data to device." << std::endl;
    return false;
  }
  state_subscription_ = this->subscribe();
  for (size_t i = 0; i < ncs::sim::Constants::num_buffers; ++i) {
    auto blank = new CalciumDependentBuffer<MType>();
    if (!blank->init(num_channels)) {
      std::cerr << "Failed to initialize CalciumDependentBuffer." << std::endl;
      delete blank;
      return false;
    }
    this->addBlank(blank);
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
