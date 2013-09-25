#pragma once

#include <ncs/spec/ModelParameters.h>

namespace ncs {

namespace spec {

class InputGroup {
public:
  InputGroup(const std::string& neuron_alias,
             ModelParameters* model_parameters,
             double probability,
             float start_time,
             float end_time);

  const std::string& getNeuronAlias() const;

  ModelParameters* getModelParameters() const;

  double getProbability() const;

  float getStartTime() const;
  float getEndTime() const;
private:
  std::string neuron_alias_;
  ModelParameters* model_parameters_;
  double probability_;
  float start_time_;
  float end_time_;
};

} // namespace spec

} // namespace ncs
