#include <ncs/sim/DataBuffer.h>
#include <ncs/sim/SynapseSimulator.h>
#include <ncs/spec/Generator.h>

struct Instantiator {
  ncs::spec::Generator* utilization;
  ncs::spec::Generator* redistribution;
  ncs::spec::Generator* last_prefire_time;
  ncs::spec::Generator* last_postfire_time;
  ncs::spec::Generator* tau_depression;
  ncs::spec::Generator* tau_facilitation;
  ncs::spec::Generator* tau_ltp;
  ncs::spec::Generator* tau_ltd;
  ncs::spec::Generator* A_ltp_minimum;
  ncs::spec::Generator* A_ltd_minimum;
  ncs::spec::Generator* max_conductance;
  ncs::spec::Generator* reversal_potential;
  ncs::spec::Generator* tau_postsynaptic_conductance;
  ncs::spec::Generator* psg_waveform_duration;
  ncs::spec::Generator* delay;
};

template<ncs::sim::DeviceType::Type MType>
class NCSDataBuffer : public ncs::sim::DataBuffer {
public:
  NCSDataBuffer();
  bool init(size_t num_synapses);
  bool clear();
  bool expandAndClear(size_t new_size);
  unsigned int maximum_size;
  unsigned int current_size;
  unsigned int* device_current_size;
  float* fire_time;
  unsigned int* fire_index;
  float* psg_max;
  virtual ~NCSDataBuffer();
};

template<>
class NCSDataBuffer<ncs::sim::DeviceType::CPU> : public ncs::sim::DataBuffer {
public:
  NCSDataBuffer();
  bool init(size_t num_synapses);
  std::vector<float> fire_time;
  std::vector<unsigned int> fire_index;
  std::vector<float> psg_max;
};

template<ncs::sim::DeviceType::Type MType>
class NCSSimulator 
  : public ncs::sim::SynapseSimulator<MType>,
    public ncs::sim::SpecificPublisher<NCSDataBuffer<MType>> {
public:
  NCSSimulator();
  virtual bool addSynapse(ncs::sim::Synapse* synapse);
  virtual bool initialize();
  virtual bool update(ncs::sim::SynapseUpdateParameters* parameters);
private:
  unsigned int* device_neuron_device_ids_;
  std::vector<ncs::sim::Synapse*> synapses_;
  float* utilization_;
  float* redistribution_;
  float* last_prefire_time_;
  float* last_postfire_time_;
  float* tau_depression_;
  float* tau_facilitation_;
  float* tau_ltp_;
  float* tau_ltd_;
  float* A_ltp_minimum_;
  float* A_ltd_minimum_;
  float* max_conductance_;
  float* reversal_potential_;
  float* tau_postsynaptic_conductance_;
  float* psg_waveform_duration_;

  float* base_utilization_;
  float* A_ltp_;
  float* A_ltd_;
  size_t num_synapses_;
  typename ncs::sim::SpecificPublisher<NCSDataBuffer<MType>>::Subscription* 
    self_subscription_;
};

template<>
bool NCSSimulator<ncs::sim::DeviceType::CPU>::
update(ncs::sim::SynapseUpdateParameters* parameters);

#ifdef NCS_CUDA
template<>
bool NCSSimulator<ncs::sim::DeviceType::CUDA>::
update(ncs::sim::SynapseUpdateParameters* parameters);
#endif // NCS_CUDA

#include "NCS.hpp"
