#include <algorithm>

#include <ncs/sim/Constants.h>
#include <ncs/sim/Memory.h>

template<ncs::sim::DeviceType::Type MType>
NeuronBuffer<MType>::NeuronBuffer() {
  calcium_ = nullptr;
  voltage_ = nullptr;
  spike_shape_state_ = nullptr;
}

template<ncs::sim::DeviceType::Type MType>
bool NeuronBuffer<MType>::init(size_t num_neurons) {
  using ncs::sim::Memory;
  if (num_neurons > 0) {
    bool result = true;
    result &= Memory<MType>::malloc(calcium_, num_neurons);
    result &= Memory<MType>::malloc(voltage_, num_neurons);
    result &= Memory<MType>::malloc(spike_shape_state_, num_neurons);
    return result;
  }
  return true;
}

template<ncs::sim::DeviceType::Type MType>
float* NeuronBuffer<MType>::getVoltage() {
  return voltage_;
}

template<ncs::sim::DeviceType::Type MType>
float* NeuronBuffer<MType>::getCalcium() {
  return calcium_;
}

template<ncs::sim::DeviceType::Type MType>
int* NeuronBuffer<MType>::getSpikeShapeState() {
  return spike_shape_state_;
}

template<ncs::sim::DeviceType::Type MType>
NeuronBuffer<MType>::~NeuronBuffer() {
  using ncs::sim::Memory;
  if (voltage_) {
    Memory<MType>::free(voltage_);
  }
  if (calcium_) {
    Memory<MType>::free(calcium_);
  }
  if (spike_shape_state_) {
    Memory<MType>::free(spike_shape_state_);
  }
}

