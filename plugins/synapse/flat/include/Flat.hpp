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
  return true;
}

template<ncs::sim::DeviceType::Type MType>
bool FlatSimulator<MType>::initialize() {
  size_t num_synapses = cpu_current_.size();
  using ncs::sim::Memory;
  bool result = true;
  result &= Memory<MType>::malloc(device_current_, num_synapses);
  if (!result) {
    std::cerr << "Failed to allocate memory." << std::endl;
    return false;
  }

  const auto CPU = ncs::sim::DeviceType::CPU;
  result &= Memory<CPU>::To<MType>::copy(cpu_current_.data(),
                                         device_current_,
                                         num_synapses);
  return result;
}
