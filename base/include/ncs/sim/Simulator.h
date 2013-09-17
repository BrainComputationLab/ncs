#pragma once

#include <ncs/sim/ClusterDescription.h>
#include <ncs/sim/Device.h>
#include <ncs/sim/FactoryMap.h>
#include <ncs/sim/ModelStatistics.h>
#include <ncs/sim/MPI.h>
#include <ncs/sim/NeuronSimulator.h>
#include <ncs/sim/PluginLoader.h>
#include <ncs/sim/SynapseSimulator.h>
#include <ncs/sim/VectorExchanger.h>
#include <ncs/spec/ModelSpecification.h>

namespace ncs {

namespace sim {

class Simulator {
public:
  Simulator(spec::ModelSpecification* model_specification);
  bool initialize(int argc, char** argv);
private:
  bool initializeSeeds_();
  bool gatherClusterData_(unsigned int enabled_device_types);
  bool gatherModelStatistics_();
  bool allocateNeurons_();
  bool distributeNeurons_();
  bool assignNeuronIDs_();
  bool loadNeuronSimulatorPlugins_();
  bool loadNeuronInstantiators_();
  bool loadSynapseSimulatorPlugins_();
  bool loadSynapseInstantiators_();
  bool distributeSynapses_();
  bool initializeDevices_();
  bool initializeVectorExchanger_();

  int getNeuronSeed_() const;
  int getSynapseSeed_() const;

  spec::ModelSpecification* model_specification_;
  ModelStatistics* model_statistics_;
  Communicator* communicator_;
  ClusterDescription* cluster_;

  FactoryMap<NeuronSimulator>* neuron_simulator_generators_;
  std::map<spec::NeuronGroup*, std::vector<Neuron*>> neurons_by_group_;
  std::map<spec::NeuronGroup*, void*> neuron_instantiators_by_group_;
  std::vector<size_t> neuron_global_id_offsets_;
  size_t global_neuron_vector_size_;
  Neuron* neurons_;
  unsigned int num_neurons_;

  FactoryMap<SynapseSimulator>* synapse_simulator_generators_;
  std::map<spec::SynapseGroup*, void*> synapse_instantiators_by_group_;
  std::map<spec::SynapseGroup*, std::vector<Synapse*>> synapses_by_group_;

  MachineVectorExchanger* vector_exchanger_;

  int neuron_seed_;
  int synapse_seed_;

  std::vector<DeviceBase*> devices_;
};

} // namespace sim

} // namespace ncs
