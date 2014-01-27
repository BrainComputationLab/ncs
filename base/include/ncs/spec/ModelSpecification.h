#pragma once

#include <ncs/spec/ModelParameters.h>
#include <ncs/spec/NeuronAlias.h>
#include <ncs/spec/NeuronGroup.h>
#include <ncs/spec/SynapseAlias.h>
#include <ncs/spec/SynapseGroup.h>

namespace ncs {

namespace spec {

struct ModelSpecification {
  std::map<std::string, NeuronGroup*> neuron_groups;
  std::map<std::string, NeuronAlias*> neuron_aliases;
  std::map<std::string, SynapseGroup*> synapse_groups;
  std::map<std::string, SynapseAlias*> synapse_aliases;
  std::map<std::string, ModelParameters*> neuron_parameters;
  std::map<std::string, ModelParameters*> synapse_parameters;
};

} // namespace spec

} // namespace ncs
