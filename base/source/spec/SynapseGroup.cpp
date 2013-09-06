#include <ncs/spec/SynapseGroup.h>

namespace ncs {

namespace spec {

SynapseGroup::
SynapseGroup(const std::vector<NeuronGroup*>& presynaptic_neurons,
             const std::vector<NeuronGroup*>& postsynaptic_neurons,
             ModelParameters* model_parameters,
             double connection_probability)
  : presynaptic_neurons_(presynaptic_neurons),
    postsynaptic_neurons_(postsynaptic_neurons),
    model_parameters_(model_parameters),
    connection_probability_(connection_probability) {
}

const std::vector<NeuronGroup*>& SynapseGroup::getPresynapticGroups() const {
  return presynaptic_neurons_;
}

const std::vector<NeuronGroup*>& SynapseGroup::getPostsynapticGroups() const {
  return postsynaptic_neurons_;
}

ModelParameters* SynapseGroup::getModelParameters() const {
  return model_parameters_;
}

double SynapseGroup::getConnectionProbability() const {
  return connection_probability_;
}

} // namespace spec

} // namespace ncs
