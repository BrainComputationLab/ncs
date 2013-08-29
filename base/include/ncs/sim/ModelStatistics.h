#pragma once

#include <ncs/spec/ModelSpecification.h>

namespace ncs {

namespace sim {

class ModelStatistics {
public:
  ModelStatistics(spec::ModelSpecification* model);
  const std::map<spec::NeuronGroup*, double>& estimateNeuronLoad() const;
  double estimateTotalLoad() const;
  unsigned int getNumberOfNeurons() const;
private:
  void analyzeModel_(spec::ModelSpecification* model);
  std::map<spec::NeuronGroup*, double> load_by_neuron_group_;
  double total_load_;
  unsigned int num_neurons_;
};

} // namespace sim

} // namespace ncs
