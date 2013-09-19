#pragma once

#include <ncs/sim/DeviceNeuronStateBuffer.h>
#include <ncs/sim/DeviceType.h>
#include <ncs/sim/InputBuffer.h>
#include <ncs/sim/SynapticCurrentBuffer.h>

namespace ncs {

namespace sim {

template<DeviceType::Type MType>
class NeuronSimulatorUpdater 
  : public SpecificPublisher<DeviceNeuronStateBuffer<MType>> {
public:
  typedef SpecificPublisher<DeviceNeuronStateBuffer<MType>>
    NeuronStatePublisher;
  typedef SpecificPublisher<InputBuffer<MType>> InputPublisher;
  typedef SpecificPublisher<SynapticCurrentBuffer<MType>>
    SynapticCurrentPublisher;
  NeuronSimulatorUpdater();
  ~NeuronSimulatorUpdater();
private:
  typename NeuronStatePublisher::Subscription* neuron_state_subscription_;
  typename InputPublisher::Subscription* input_subscription_;
  typename SynapticCurrentPublisher::Subscription* 
    synaptic_current_subscription_;
};

} // namespace sim

} // namespace ncs

#include <ncs/sim/NeuronSimulatorUpdater.hpp>
