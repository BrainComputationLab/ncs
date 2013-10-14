#include <ncs/sim/SynapseSimulator.h>
#include <ncs/spec/Generator.h>

struct Instantiator {
  ncs::spec::Generator* delay;
  ncs::spec::Generator* current;
};

template<ncs::sim::DeviceType::Type MType>
class FlatSimulator : public ncs::sim::SynapseSimulator<MType> {
public:
  FlatSimulator();
  virtual bool addSynapse(ncs::sim::Synapse* synapse);
  virtual bool initialize();
  virtual bool update(ncs::sim::SynapseUpdateParameters* parameters);
private:
  std::vector<float> cpu_current_;
  std::vector<unsigned int> cpu_neuron_device_ids_;
  float* device_current_;
  unsigned int* device_neuron_device_ids_;
  size_t num_synapses_;
};

template<>
bool FlatSimulator<ncs::sim::DeviceType::CPU>::
update(ncs::sim::SynapseUpdateParameters* parameters);

template<>
bool FlatSimulator<ncs::sim::DeviceType::CUDA>::
update(ncs::sim::SynapseUpdateParameters* parameters);

#include "Flat.hpp"
