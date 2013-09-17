namespace ncs {

namespace sim {

template<DeviceType::Type MType>
FireTableUpdater<MType>::FireTableUpdater()
  : fire_table_(nullptr),
    global_presynaptic_neuron_ids_(nullptr),
    synaptic_delays_(nullptr) {
}

template<DeviceType::Type MType>
bool FireTableUpdater<MType>::
init(FireTable<MType>* table,
     SpecificPublisher<GlobalNeuronStateBuffer<MType>>* publisher,
     const std::vector<Synapse*> synapse_vector) {
  fire_table_ = table;
  std::vector<unsigned int> global_presynaptic_neuron_ids;
  std::vector<unsigned int> synaptic_delays;
  for (auto synapse : synapse_vector) {
    if (nullptr == synapse) {
      const unsigned int null_value = std::numeric_limits<unsigned int>::max();
      global_presynaptic_neuron_ids.push_back(null_value);
      synaptic_delays.push_back(null_value);
    } else {
      unsigned int global_id = synapse->presynaptic_neuron->id.global;
      global_presynaptic_neuron_ids.push_back(global_id);
      synaptic_delays.push_back(synapse->delay);
    }
  }
  bool result = true;
  result &= Memory<MType>::malloc(global_presynaptic_neuron_ids_,
                                  global_presynaptic_neuron_ids.size());
  result &= Memory<MType>::malloc(synaptic_delays_, synaptic_delays.size());
  if (!result) {
    std::cerr << "Failed to allocate synaptic data arrays." << std::endl;
    return false;
  }
  result &= 
    mem::copy<MType, DeviceType::CPU>(global_presynaptic_neuron_ids_,
                                      global_presynaptic_neuron_ids.data(),
                                      global_presynaptic_neuron_ids.size());
  result &=
    mem::copy<MType, DeviceType::CPU>(synaptic_delays_,
                                      synaptic_delays.data(),
                                      synaptic_delays.size());
  if (!result) {
    std::cerr << "Failed to copy synaptic data arrays." << std::endl;
    return false;
  }
  device_synaptic_vector_size_ = synapse_vector.size();
  subscription_ = publisher->subscribe();
  return nullptr != subscription_;
}

template<DeviceType::Type MType>
FireTableUpdater<MType>::~FireTableUpdater() {
  if (global_presynaptic_neuron_ids_) {
    Memory<MType>::free(global_presynaptic_neuron_ids_);
  }
  if (synaptic_delays_) {
    Memory<MType>::free(synaptic_delays_);
  }
  if (subscription_) {
    delete subscription_;
  }
}

} // namespace sim

} // namespace ncs
