#include <ncs/spec/InputGroup.h>
#include "ModelParameters.pb.h"

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

bool InputGroup::toProtobuf(com::InputGroup* input_group) const {
  for (auto alias : neuron_aliases_) {
    input_group->add_neuron_alias(alias);
  }
  model_parameters_->makeProtobuf(input_group->mutable_model_parameters());
  input_group->set_probability(probability_);
  input_group->set_start_time(start_time_);
  input_group->set_end_time(end_time_);
  return true;
}

InputGroup* InputGroup::fromProtobuf(com::InputGroup* input_group) {
  std::vector<std::string> neuron_aliases;
  for (int i = 0; i < input_group->neuron_alias_size(); ++i) {
    neuron_aliases.push_back(input_group->neuron_alias(i));
  }
  ModelParameters* mp = 
    ModelParameters::fromProtobuf(input_group->mutable_model_parameters());
  return new InputGroup(neuron_aliases,
                        mp,
                        input_group->probability(),
                        input_group->start_time(),
                        input_group->end_time());
}

} // namespace spec

} // namespace ncs
