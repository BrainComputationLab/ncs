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
  bool init(InputPublisher* input_publisher,
            SynapticCurrentPublisher* synaptic_current_publisher,
            const std::vector<NeuronSimulator<MType>*>& neuron_simulators,
            const std::vector<size_t>& device_id_offsets,
            size_t neuron_device_vector_size,
            size_t num_buffers);
  bool start();
  ~NeuronSimulatorUpdater();
private:
  typename NeuronStatePublisher::Subscription* neuron_state_subscription_;
  typename InputPublisher::Subscription* input_subscription_;
  typename SynapticCurrentPublisher::Subscription* 
    synaptic_current_subscription_;
  std::vector<NeuronSimulator<MType>*> neuron_simulators_;
  std::vector<size_t> device_id_offsets_;
  size_t neuron_device_vector_size_;
  size_t num_buffers_;

  std::thread master_thread_;
  std::vector<std::thread> worker_threads_;
};

} // namespace sim

} // namespace ncs

#include <ncs/sim/NeuronSimulatorUpdater.hpp>
