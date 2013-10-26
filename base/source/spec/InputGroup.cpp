#include <ncs/spec/InputGroup.h>

namespace ncs {

namespace spec {

InputGroup::InputGroup(const std::vector<std::string>& neuron_aliases,
                       ModelParameters* model_parameters,
                       double probability,
                       float start_time,
                       float end_time)
  : neuron_aliases_(neuron_aliases),
    model_parameters_(model_parameters),
    probability_(probability),
    start_time_(start_time),
    end_time_(end_time) {
}

const std::vector<std::string>& InputGroup::getNeuronAliases() const {
  return neuron_aliases_;
}

ModelParameters* InputGroup::getModelParameters() const {
  return model_parameters_;
}

double InputGroup::getProbability() const {
  return probability_;
}

float InputGroup::getStartTime() const {
  return start_time_;
}

float InputGroup::getEndTime() const {
  return end_time_;
}

} // namespace spec

} // namespace ncs
