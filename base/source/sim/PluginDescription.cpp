#include <ncs/sim/PluginDescription.h>

namespace ncs {

namespace sim {

NeuronPluginDescription::
NeuronPluginDescription(const std::string& model_type)
  : model_type_(model_type) {
}

unsigned int NeuronPluginDescription::addNeuron(Neuron* neuron) {
  neurons_.push_back(neuron);
  return neurons_.size() - 1;
}

const std::vector<Neuron*>&
NeuronPluginDescription::getNeurons() const {
  return neurons_;
}

const std::string& NeuronPluginDescription::getType() const {
  return model_type_;
}

SynapsePluginDescription::
SynapsePluginDescription(const std::string& model_type)
  : model_type_(model_type) {
}

unsigned int SynapsePluginDescription::addSynapse(Synapse* synapse) {
  synapses_.push_back(synapse);
  return synapses_.size() - 1;
}

const std::vector<Synapse*>&
SynapsePluginDescription::getSynapses() const {
  return synapses_;
}

const std::string& SynapsePluginDescription::getType() const {
  return model_type_;
}

} // namespace sim

} // namespace ncs
