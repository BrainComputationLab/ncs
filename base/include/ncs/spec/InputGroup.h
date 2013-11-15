#pragma once
#include <ncs/spec/ModelParameters.h>

namespace ncs {

namespace com {
  struct InputGroup;
}

namespace spec {

class InputGroup {
public:
  InputGroup(const std::vector<std::string>& neuron_alias,
             ModelParameters* model_parameters,
             double probability,
             float start_time,
             float end_time);

  const std::vector<std::string>& getNeuronAliases() const;

  ModelParameters* getModelParameters() const;

  double getProbability() const;

  float getStartTime() const;
  float getEndTime() const;
  bool toProtobuf(com::InputGroup* input_group) const;
  static InputGroup* fromProtobuf(com::InputGroup* input_group);
private:
  std::vector<std::string> neuron_aliases_;
  ModelParameters* model_parameters_;
  double probability_;
  float start_time_;
  float end_time_;
};

} // namespace spec

} // namespace ncs
