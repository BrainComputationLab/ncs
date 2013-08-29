#pragma once
#include <string>
#include <vector>

#include <ncs/sim/Neuron.h>

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

} // namespace sim

} // namespace ncs
