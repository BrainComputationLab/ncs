#include <ncs/sim/FireTableUpdater.h>
#ifdef NCS_CUDA
#include <ncs/cuda/FireTableUpdater.cuh>
#endif // NCS_CUDA

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

#ifdef NCS_CUDA
template<>
bool FireTableUpdater<DeviceType::CUDA>::
update_(GlobalNeuronStateBuffer<DeviceType::CUDA>* neuron_state,
        unsigned int step) {
  cuda::updateFireTable(neuron_state->getFireBits(),
                        fire_table_->getTable(),
                        fire_table_->getWordsPerVector(),
                        step % fire_table_->getNumberOfRows(),
                        fire_table_->getNumberOfRows(),
                        global_presynaptic_neuron_ids_,
                        synaptic_delays_,
                        device_synaptic_vector_size_);
  return true;
}
#endif // NCS_CUDA

} // namespace sim

} // namespace ncs
