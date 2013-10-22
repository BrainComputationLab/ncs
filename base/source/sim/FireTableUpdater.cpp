#include <ncs/sim/FireTableUpdater.h>

namespace ncs {

namespace sim {

template<>
bool FireTableUpdater<DeviceType::CPU>::
update_(GlobalNeuronStateBuffer<DeviceType::CPU>* neuron_state,
        unsigned int step) {
  const Bit::Word* fire_vector = neuron_state->getFireBits();
  for (size_t i = 0; i < device_synaptic_vector_size_; ++i) {
    unsigned int presynaptic_neuron_id = global_presynaptic_neuron_ids_[i];
    if (presynaptic_neuron_id == 0xFFFFFFFF) {
      continue;
    }
    unsigned int word_index = Bit::word(presynaptic_neuron_id);
    Bit::Word mask = Bit::mask(presynaptic_neuron_id);
    if (fire_vector[word_index] & mask) {
      unsigned int delay = synaptic_delays_[i];
      unsigned int event_row = step + delay;
      Bit::Word* synaptic_fire_vector = fire_table_->getRow(event_row);
      Bit::Word event_mask = Bit::mask(i);
      synaptic_fire_vector[Bit::word(i)] |= event_mask;
    }
  }
  return true;
}


template<>
bool FireTableUpdater<DeviceType::CUDA>::
update_(GlobalNeuronStateBuffer<DeviceType::CUDA>* neuron_state,
        unsigned int step) {
  std::cout << "STUB: FireTableUpdater<CUDA>::update_()" << std::endl;
  return true;
}


} // namespace sim

} // namespace ncs
