#pragma once

#include <ncs/sim/ClusterDescription.h>
#include <ncs/sim/ModelStatistics.h>
#include <ncs/sim/MPI.h>
#include <ncs/sim/NeuronSimulator.h>
#include <ncs/sim/PluginLoader.h>
#include <ncs/spec/ModelSpecification.h>

namespace ncs {

namespace sim {

class Simulator {
public:
  Simulator(spec::ModelSpecification* model_specification);
  bool initialize(int argc, char** argv);
private:
  bool gatherClusterData_(unsigned int enabled_device_types);
  bool gatherModelStatistics_();
  bool allocateNeurons_();
  bool distributeNeurons_();
  bool assignNeuronIDs_();
  bool loadNeuronSimulatorPlugins_();
  bool loadNeuronInstantiators_();

  spec::ModelSpecification* model_specification_;
  ModelStatistics* model_statistics_;
  Communicator* communicator_;
  ClusterDescription* cluster_;
  FactoryMap<NeuronSimulator>* neuron_simulator_generators_;
  std::map<spec::NeuronGroup*, std::vector<Neuron*>> neurons_by_group_;
  std::map<spec::NeuronGroup*, void*> instantiators_by_group_;
  Neuron* neurons_;
  unsigned int num_neurons_;
};

} // namespace sim

} // namespace ncs
