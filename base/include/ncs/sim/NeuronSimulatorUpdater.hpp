namespace ncs {

namespace sim {

template<DeviceType::Type MType>
NeuronSimulatorUpdater<MType>::NeuronSimulatorUpdater()
  : neuron_state_subscription_(nullptr),
    input_subscription_(nullptr),
    synaptic_current_subscription_(nullptr) {
}

template<DeviceType::Type MType>
NeuronSimulatorUpdater<MType>::~NeuronSimulatorUpdater() {
  if (neuron_state_subscription_) {
    delete neuron_state_subscription_;
  }
  if (input_subscription_) {
    delete input_subscription_;
  }
  if (synaptic_current_subscription_) {
    delete synaptic_current_subscription_;
  }
}

} // namespace sim

} // namespace ncs
