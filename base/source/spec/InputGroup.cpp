#include <ncs/spec/InputGroup.h>

namespace ncs {

namespace spec {

InputGroup::InputGroup(const std::string& neuron_alias,
                       ModelParameters* model_parameters,
                       double probability)
  : neuron_alias_(neuron_alias),
    model_parameters_(model_parameters),
    probability_(probability) {
}

const std::string& InputGroup::getNeuronAlias() const {
  return neuron_alias_;
}

ModelParameters* InputGroup::getModelParameters() const {
  return model_parameters_;
}

double InputGroup::getProbability() const {
  return probability_;
}

} // namespace spec

} // namespace ncs
