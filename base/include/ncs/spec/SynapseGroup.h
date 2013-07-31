#pragma once

#include <ncs/spec/NeuronGroup.h>

namespace ncs {

namespace spec {

/**
  Represents a set of connections from one set of neurons to another set
  of neurons.
*/
class SynapseGroup {
public:
  /**
    Constructor.

    @param presynaptic_neurons The neurons on the transmitting end.
    @param postsynaptic_neurons The neurons on the receiving end.
    @param model_parameters The parameters used to generate the synapses.
    @param connection_probability The chance that there will be a connection
      between the presynaptic neuron to a postsynaptic one.
  */
  SynapseGroup(const std::vector<NeuronGroup*>& presynaptic_neurons,
               const std::vector<NeuronGroup*>& postsynaptic_neurons,
               ModelParameters* model_parameters,
               double connection_probability);

  /**
    Returns the transmitting neuron group.
  
    @return A list of neuron groups on the transmitting end.
  */
  const std::vector<NeuronGroup*>& getPresynapticGroups() const;

  /**
    Returns the receiving neuron group.

    @return A list of neuron groups on the receiving end.
  */
  const std::vector<NeuronGroup*>& getPostsynapticGroups() const;

  /**
    Returns the parameters used to generate the synaptic model.

    @return The parameters used to generate the synaptic model.
  */
  ModelParameters* getModelParameters() const;

  /**
    Returns the probability that a connection will exist from a presynaptic
    neuron to a postsynaptic one due to this synapse specification.

    @return The connection probability.
  */
  double getConnectionProbability() const;
private:
  /// The group of neurons that transmit a signal across these synapses
  std::vector<NeuronGroup*> presynaptic_neurons_;

  /// The group of neurons that receive a signal across these synapses
  std::vector<NeuronGroup*> postsynaptic_neurons_;

  /// The parameters used to specify the synaptic model
  ModelParameters* model_parameters_;

  /// The probability of connection between the pre and post synaptic groups
  double connection_probability_;
};

} // namespace spec

} // namespace ncs
