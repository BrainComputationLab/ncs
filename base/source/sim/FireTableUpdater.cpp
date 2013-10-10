#include <ncs/sim/FireTableUpdater.h>

namespace ncs {

namespace sim {

template<>
bool FireTableUpdater<DeviceType::CPU>::
update_(GlobalNeuronStateBuffer<DeviceType::CPU>* neuron_state,
        unsigned int step) {
  std::cout << "STUB: FireTableUpdater<CPU>::update_()" << std::endl;
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
