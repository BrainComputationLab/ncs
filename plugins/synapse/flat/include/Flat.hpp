#include <ncs/sim/Memory.h>

template<ncs::sim::DeviceType::Type MType>
FlatSimulator<MType>::FlatSimulator()
  : device_current_(nullptr) {
}

template<ncs::sim::DeviceType::Type MType>
bool FlatSimulator<MType>::addSynapse(ncs::sim::Synapse* synapse) {
  Instantiator* i = (Instantiator*)(synapse->instantiator);
  ncs::spec::RNG rng(synapse->seed);
  cpu_current_.push_back(i->current->generateDouble(&rng));
  synapse->delay = i->delay->generateInt(&rng);
  cpu_neuron_device_ids_.push_back(synapse->postsynaptic_neuron->id.device);
  return true;
}

template<ncs::sim::DeviceType::Type MType>
bool FlatSimulator<MType>::initialize() {
  num_synapses_ = cpu_current_.size();
  using ncs::sim::Memory;
  bool result = true;
  result &= Memory<MType>::malloc(device_current_, num_synapses_);
  result &= Memory<MType>::malloc(device_neuron_device_ids_, num_synapses_);
  if (!result) {
    std::cerr << "Failed to allocate memory." << std::endl;
    return false;
  }

  const auto CPU = ncs::sim::DeviceType::CPU;
  result &= Memory<CPU>::To<MType>::copy(cpu_current_.data(),
                                         device_current_,
                                         num_synapses_);
  result &= Memory<CPU>::To<MType>::copy(cpu_neuron_device_ids_.data(),
                                         device_neuron_device_ids_,
                                         num_synapses_);
  if (!result) {
    std::cerr << "Failed to transfer memory from CPU to " << 
      ncs::sim::DeviceType::as_string(MType) << std::endl;
    return false;
  }
  return result;
}

template<ncs::sim::DeviceType::Type MType>
bool FlatSimulator<MType>::
update(ncs::sim::SynapseUpdateParameters* parameters) {
  std::clog << "STUB: FlatSimulator::update()" << std::endl;
  return true;
}
