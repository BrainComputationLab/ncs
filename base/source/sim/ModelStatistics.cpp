#include <ncs/sim/ModelStatistics.h>

namespace ncs {

namespace sim {

ModelStatistics::
ModelStatistics(spec::ModelSpecification* model) {
  analyzeModel_(model);
}

const std::map<spec::NeuronGroup*, double>&
ModelStatistics::estimateNeuronLoad() const {
  return load_by_neuron_group_;
}

double ModelStatistics::estimateTotalLoad() const {
  return total_load_;
}

unsigned int ModelStatistics::getNumberOfNeurons() const {
  return num_neurons_;
}

void ModelStatistics::analyzeModel_(spec::ModelSpecification* model) {
  // Estimate load per neuron
  for (auto& key_value : model->neuron_groups) {
    spec::NeuronGroup* neuron_group = key_value.second;
    load_by_neuron_group_[neuron_group] = 0.0;
  }
  for (auto& key_value : model->synapse_groups) {
    spec::SynapseGroup* synapse_group = key_value.second;
    const auto& presynaptic_groups = synapse_group->getPresynapticGroups();
    unsigned int num_presynaptic_neurons = 0;
    for (auto group : presynaptic_groups) {
      num_presynaptic_neurons += group->getNumberOfCells();
    }
    double probability = synapse_group->getConnectionProbability();
    double connections_per_neuron = num_presynaptic_neurons * probability;
    const auto& postsynaptic_groups = synapse_group->getPostsynapticGroups();
    // TODO(rvhoang): Consider weighting based on the type of synapse as well
    // Right now, all synapse types are weighted equally
    for (auto group : postsynaptic_groups) {
      load_by_neuron_group_[group] += connections_per_neuron;
    }
  }

  // Sum the total load
  total_load_ = 0.0;
  for (auto key_value : load_by_neuron_group_) {
    spec::NeuronGroup* group = key_value.first;
    double load_per_cell = key_value.second;
    total_load_ += load_per_cell * group->getNumberOfCells();
  }

  // Get the total number of neurons
  num_neurons_ = 0;
  for (auto key_value : model->neuron_groups) {
    spec::NeuronGroup* group = key_value.second;
    num_neurons_ += group->getNumberOfCells();
  }
}

} // namespace sim

} // namespace ncs
