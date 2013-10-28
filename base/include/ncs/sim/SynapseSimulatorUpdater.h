#pragma once

#include <ncs/sim/DeviceNeuronStateBuffer.h>
#include <ncs/sim/SynapticCurrentBuffer.h>
#include <ncs/sim/SynapticFireVectorBuffer.h>
#include <ncs/sim/SynapseSimulator.h>

namespace ncs {

namespace sim {

template<DeviceType::Type MType>
class SynapseSimulatorUpdater 
  : public SpecificPublisher<SynapticCurrentBuffer<MType>> {
public:
  typedef SynapticFireVectorBuffer<MType> FireVectorBuffer;
  typedef SpecificPublisher<FireVectorBuffer> FireVectorPublisher;
  typedef DeviceNeuronStateBuffer<MType> NeuronStateBuffer;
  typedef SpecificPublisher<NeuronStateBuffer> NeuronStatePublisher;

  SynapseSimulatorUpdater();
  bool setFireVectorPublisher(FireVectorPublisher* publisher);
  bool setNeuronStatePublisher(NeuronStatePublisher* publisher);
  bool init(const std::vector<SynapseSimulator<MType>*>& simulators,
            const std::vector<size_t>& device_synaptic_vector_offsets,
            const spec::SimulationParameters* simulation_parameters,
            size_t neuron_device_vector_size,
            size_t num_buffers);
  bool start();
  ~SynapseSimulatorUpdater();
private:
  typename FireVectorPublisher::Subscription* fire_subscription_;
  typename NeuronStatePublisher::Subscription* neuron_state_subscription_;
  std::vector<SynapseSimulator<MType>*> simulators_;
  std::vector<size_t> device_synaptic_vector_offsets_;
  size_t neuron_device_vector_size_;
  size_t num_buffers_;
  std::thread master_thread_;
  std::vector<std::thread> worker_threads_;
  const spec::SimulationParameters* simulation_parameters_;
};

} // namespace sim

} // namespace ncs

#include <ncs/sim/SynapseSimulatorUpdater.hpp>
