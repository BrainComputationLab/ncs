#pragma once

#include <ncs/spec/ModelParameters.h>

namespace ncs {

namespace spec {

class InputGroup {
public:
  InputGroup(const std::string& neuron_alias,
             ModelParameters* model_parameters,
             double probability);

  const std::string& getNeuronAlias() const;

  ModelParameters* getModelParameters() const;

  double getProbability() const;
private:
  std::string neuron_alias_;
  ModelParameters* model_parameters_;
  double probability_;
};

} // namespace spec

} // namespace ncs
