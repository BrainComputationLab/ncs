#pragma once
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
  ~FireTableUpdater();
private:
  unsigned int* global_presynaptic_neuron_ids_;
  unsigned int* synaptic_delays_;
  size_t device_synaptic_vector_size_;
  SpecificPublisher<GlobalNeuronStateBuffer<MType>>* subscription_;
};

} // namespace sim

} // namespace ncs

#include <ncs/sim/FireTableUpdater.h>
