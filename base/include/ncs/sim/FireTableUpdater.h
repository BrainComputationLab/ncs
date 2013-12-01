#pragma once

#include <thread>

#include <ncs/sim/DataBuffer.h>
#include <ncs/sim/FireTable.h>
#include <ncs/sim/GlobalNeuronStateBuffer.h>
#include <ncs/sim/Synapse.h>
#include <ncs/sim/SynapticFireVectorBuffer.h>

namespace ncs {

namespace sim {

template<DeviceType::Type MType>
class FireTableUpdater 
  : public SpecificPublisher<SynapticFireVectorBuffer<MType>> {
public:
  FireTableUpdater();
  bool init(FireTable<MType>* table,
            SpecificPublisher<GlobalNeuronStateBuffer<MType>>* publisher,
            const std::vector<Synapse*> synapse_vector);
  bool start(std::function<bool()> thread_init,
             std::function<bool()> thread_destroy);
  ~FireTableUpdater();
private:
  bool update_(GlobalNeuronStateBuffer<MType>* neuron_state,
               unsigned int step);

  unsigned int* global_presynaptic_neuron_ids_;
  unsigned int* synaptic_delays_;
  size_t device_synaptic_vector_size_;
  typedef SpecificPublisher<GlobalNeuronStateBuffer<MType>>
    GlobalNeuronStatePublisher;
  typename GlobalNeuronStatePublisher::Subscription* subscription_;
  FireTable<MType>* fire_table_;
  std::thread thread_;
};

template<>
bool FireTableUpdater<DeviceType::CPU>::
update_(GlobalNeuronStateBuffer<DeviceType::CPU>* neuron_state,
        unsigned int step);

template<>
bool FireTableUpdater<DeviceType::CUDA>::
update_(GlobalNeuronStateBuffer<DeviceType::CUDA>* neuron_state,
        unsigned int step);

} // namespace sim

} // namespace ncs

#include <ncs/sim/FireTableUpdater.hpp>
