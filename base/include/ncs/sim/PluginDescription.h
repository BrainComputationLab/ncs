#pragma once
#include <string>
#include <vector>

#include <ncs/sim/Neuron.h>
#include <ncs/sim/Synapse.h>

namespace ncs {

namespace sim {

class NeuronPluginDescription {
public:
  NeuronPluginDescription(const std::string& model_type);
  unsigned int addNeuron(Neuron* neuron);
  const std::vector<Neuron*>& getNeurons() const;
  const std::string& getType() const;
private:
  std::string model_type_;
  std::vector<Neuron*> neurons_;
};

class SynapsePluginDescription {
public:
  SynapsePluginDescription(const std::string& model_type);
  unsigned int addSynapse(Synapse* synapse);
  const std::vector<Synapse*>& getSynapses() const;
  const std::string& getType() const;
private:
  std::string model_type_;
  std::vector<Synapse*> synapses_;
};

} // namespace sim

} // namespace ncs
