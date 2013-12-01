#include <ncs/sim/Constants.h>
#include <ncs/sim/Memory.h>

template<ncs::sim::DeviceType::Type MType>
NCSDataBuffer<MType>::NCSDataBuffer() {
  maximum_size = 0;
  current_size = 0;
  device_current_size = nullptr;
  fire_time = nullptr;
  fire_index = nullptr;
  psg_max = nullptr;
}

template<ncs::sim::DeviceType::Type MType>
bool NCSDataBuffer<MType>::init(size_t num_synapses) {
  maximum_size = num_synapses;
  using ncs::sim::Memory;
  bool result = true;
  result &= Memory<MType>::malloc(device_current_size, 1);
  result &= Memory<MType>::malloc(fire_time, maximum_size);
  result &= Memory<MType>::malloc(fire_index, maximum_size);
  result &= Memory<MType>::malloc(psg_max, maximum_size);
  return result;
}

template<ncs::sim::DeviceType::Type MType>
NCSDataBuffer<MType>::~NCSDataBuffer() {
  using ncs::sim::Memory;
  Memory<MType>::free(device_current_size);
  Memory<MType>::free(fire_time);
  Memory<MType>::free(fire_index);
  Memory<MType>::free(psg_max);
}

template<ncs::sim::DeviceType::Type MType>
NCSSimulator<MType>::NCSSimulator() {
  self_subscription_ = nullptr;
  utilization_ = nullptr;
  redistribution_ = nullptr;
  last_prefire_time_ = nullptr;
  last_postfire_time_ = nullptr;
  tau_depression_ = nullptr;
  tau_facilitation_ = nullptr;
  tau_ltp_ = nullptr;
  tau_ltd_ = nullptr;
  A_ltp_minimum_ = nullptr;
  A_ltd_minimum_ = nullptr;
  max_conductance_ = nullptr;
  reversal_potential_ = nullptr;
  tau_postsynaptic_conductance_ = nullptr;
  psg_waveform_duration_ = nullptr;
  base_utilization_ = nullptr;
}

template<ncs::sim::DeviceType::Type MType>
bool NCSSimulator<MType>::addSynapse(ncs::sim::Synapse* synapse) {
  synapses_.push_back(synapse);
  ncs::spec::RNG rng(synapse->seed);
  Instantiator* i = (Instantiator*)(synapse->instantiator);
  synapse->delay = i->delay->generateInt(&rng);
  return true;
}

template<ncs::sim::DeviceType::Type MType>
bool NCSSimulator<MType>::initialize() {
  using ncs::sim::Memory;
  const auto CPU = ncs::sim::DeviceType::CPU;
  num_synapses_ = synapses_.size();
  bool result = true;
  result &= Memory<MType>::malloc(device_neuron_device_ids_, num_synapses_);
  result &= Memory<MType>::malloc(utilization_, num_synapses_);
  result &= Memory<MType>::malloc(redistribution_, num_synapses_);
  result &= Memory<MType>::malloc(last_prefire_time_, num_synapses_);
  result &= Memory<MType>::malloc(last_postfire_time_, num_synapses_);
  result &= Memory<MType>::malloc(tau_depression_, num_synapses_);
  result &= Memory<MType>::malloc(tau_facilitation_, num_synapses_);
  result &= Memory<MType>::malloc(tau_ltp_, num_synapses_);
  result &= Memory<MType>::malloc(tau_ltd_, num_synapses_);
  result &= Memory<MType>::malloc(A_ltp_minimum_, num_synapses_);
  result &= Memory<MType>::malloc(A_ltd_minimum_, num_synapses_);
  result &= Memory<MType>::malloc(max_conductance_, num_synapses_);
  result &= Memory<MType>::malloc(reversal_potential_, num_synapses_);
  result &= Memory<MType>::malloc(tau_postsynaptic_conductance_, num_synapses_);
  result &= Memory<MType>::malloc(psg_waveform_duration_, num_synapses_);
  result &= Memory<MType>::malloc(base_utilization_, num_synapses_);
  result &= Memory<MType>::malloc(A_ltp_, num_synapses_);
  result &= Memory<MType>::malloc(A_ltd_, num_synapses_);
  if (!result) {
    std::cerr << "Failed to allocate memory." << std::endl;
    return false;
  }

  float* utilization = new float[num_synapses_];
  float* redistribution = new float[num_synapses_];
  float* last_prefire_time = new float[num_synapses_];
  float* last_postfire_time = new float[num_synapses_];
  float* tau_depression = new float[num_synapses_];
  float* tau_facilitation = new float[num_synapses_];
  float* tau_ltp = new float[num_synapses_];
  float* tau_ltd = new float[num_synapses_];
  float* A_ltp_minimum = new float[num_synapses_];
  float* A_ltd_minimum = new float[num_synapses_];
  float* max_conductance = new float[num_synapses_];
  float* reversal_potential = new float[num_synapses_];
  float* tau_postsynaptic_conductance = new float[num_synapses_];
  float* psg_waveform_duration = new float[num_synapses_];
  float* A_ltp = new float[num_synapses_];
  float* A_ltd = new float[num_synapses_];
  unsigned int* device_neuron_device_ids = new unsigned int[num_synapses_];
  for (size_t i = 0; i < num_synapses_; ++i) {
    Instantiator* in = (Instantiator*)(synapses_[i]->instantiator);
    ncs::spec::RNG rng(synapses_[i]->seed);
    rng();
    utilization[i] = in->utilization->generateDouble(&rng);
    redistribution[i] = in->redistribution->generateDouble(&rng);
    last_prefire_time[i] = in->last_prefire_time->generateDouble(&rng);
    last_postfire_time[i] = in->last_postfire_time->generateDouble(&rng);
    tau_depression[i] = in->tau_depression->generateDouble(&rng);
    tau_facilitation[i] = in->tau_facilitation->generateDouble(&rng);
    tau_ltp[i] = in->tau_ltp->generateDouble(&rng);
    tau_ltd[i] = in->tau_ltd->generateDouble(&rng);
    A_ltp_minimum[i] = in->A_ltp_minimum->generateDouble(&rng);
    A_ltd_minimum[i] = in->A_ltd_minimum->generateDouble(&rng);
    max_conductance[i] = in->max_conductance->generateDouble(&rng);
    reversal_potential[i] = in->reversal_potential->generateDouble(&rng);
    tau_postsynaptic_conductance[i] = 
      in->tau_postsynaptic_conductance->generateDouble(&rng);
    psg_waveform_duration[i] = in->psg_waveform_duration->generateDouble(&rng);
  }
  auto copy = [num_synapses_](float* src, float* dst) {
    return Memory<CPU>::To<MType>::copy(src, dst, num_synapses_);
  };
  result &= copy(utilization, utilization_);
  result &= copy(redistribution, redistribution_);
  result &= copy(last_prefire_time, last_prefire_time_);
  result &= copy(last_postfire_time, last_postfire_time_);
  result &= copy(tau_depression, tau_depression_);
  result &= copy(tau_facilitation, tau_facilitation_);
  result &= copy(tau_ltp, tau_ltp_);
  result &= copy(tau_ltd, tau_ltd_);
  result &= copy(A_ltp_minimum, A_ltp_minimum_);
  result &= copy(A_ltd_minimum, A_ltd_minimum_);
  result &= copy(max_conductance, max_conductance_);
  result &= copy(reversal_potential, reversal_potential_);
  result &= copy(tau_postsynaptic_conductance, tau_postsynaptic_conductance_);
  result &= copy(psg_waveform_duration, psg_waveform_duration_);
  result &= copy(utilization, base_utilization_);
  for (size_t i = 0; i < num_synapses_; ++i) {
    A_ltp[i] = 0.0;
    A_ltd[i] = 0.0;
    device_neuron_device_ids[i] = synapses_[i]->postsynaptic_neuron->id.device;
  }
  result &= copy(A_ltp, A_ltp_);
  result &= copy(A_ltd, A_ltd_);
  result &= Memory<CPU>::To<MType>::copy(device_neuron_device_ids,
                                         device_neuron_device_ids_,
                                         num_synapses_);
  if (!result) {
    std::cerr << "Failed to transfer memory from CPU to " << 
      ncs::sim::DeviceType::as_string(MType) << std::endl;
    return false;
  }

  delete [] utilization;
  delete [] redistribution;
  delete [] last_prefire_time;
  delete [] last_postfire_time;
  delete [] tau_depression;
  delete [] tau_facilitation;
  delete [] tau_ltp;
  delete [] tau_ltd;
  delete [] A_ltp_minimum;
  delete [] A_ltd_minimum;
  delete [] max_conductance;
  delete [] reversal_potential;
  delete [] tau_postsynaptic_conductance;
  delete [] psg_waveform_duration;
  delete [] A_ltp;
  delete [] A_ltd;

  synapses_.clear();
  std::vector<ncs::sim::Synapse*>().swap(synapses_);
  for (size_t i = 0; i < ncs::sim::Constants::num_buffers; ++i) {
    auto blank = new NCSDataBuffer<MType>(); 
    if (!blank->init(num_synapses_)) {
      delete blank;
      std::cerr << "Failed to initialize NCSDataBuffer." << std::endl;
      return false;
    }
    addBlank(blank);
  }

  self_subscription_ = this->subscribe();
  auto blank = this->getBlank();
  this->publish(blank);
  return true;
}
