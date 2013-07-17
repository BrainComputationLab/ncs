#pragma once

#include <Specification.h>
#include <NeuronSpecification.h>

namespace slug {

namespace spec {

// Forward declarations and container typedefs
class SynapseSpecification;
typedef std::vector<SynapseSpecification*> SynapseSpecificationList;

class SynapseGroup;
typedef std::vector<SynapseGroup*> SynapseGroupList;

class SynapseCluster;
typedef std::vector<SynapseCluster*> SynapseClusterList;

/**
  Abstract base class for a group of synapses.
*/
class SynapseGroup {
public:
  /**
    Constructor.

    @param name The name of this group.
  */
  SynapseGroup(const std::string& name);

  /**
    Returns all the SynapseSpecifications contained within this group.

    @return A list of all SynapseSpecifications contained in this group.
  */
  virtual void getSpecifications(SynapseSpecificationList* specs);

  /**
    Returns the direct subgroups that compose this SynapseGroup if any.

    @return A list of the direct subgroups that compose this SynapseGroup.
  */
  virtual SynapseGroupList getSubgroups() const;
private:
  // The name of this group.
  std::string name_;
};

/**
  Connects two groups of Neurons.
*/
class SynapseSpecification : public SynapseGroup {
public:
  /**
    Constructor.

    @param type The type of Synapse to instantiate.
    @param name The name of this SynapseGroup.
    @param presynaptic_group The group of neurons that affect the synapses.
    @param postsynaptic_group The group of neurons affected by the synapses.
    @param connection_probability The probability that a connection between two
      neurons will occur.
    @param specification Parameters used to generate the connection.
  */
  SynapseSpecification(const std::string& type,
                       const std::string& name,
                       NeuronGroup* presynaptic_group,
                       NeuronGroup* postsynaptic_group,
                       double connection_probability,
                       const Specification& specification);

  /**
    Returns all the SynapseSpecifications contained within this group.

    @return A list of all SynapseSpecifications contained in this group.
  */
  virtual void getSpecifications(SynapseSpecificationList* specs);
  
  /**
    Returns the NeuronGroup that triggers the synapses in this group.
  
    @return The presynaptic NeuronGroup.
  */
  NeuronGroup* getPresynapticGroup() const;

  /**
    returns the NeuronGroup affected by the synapses in this group.

    @return The postsynaptic NeuronGroup.
  */
  NeuronGroup* getPostsynapticGroup() const;

  /**
    Returns the probability that there will be a connection between the
    presynaptic and postsynaptic NeuronGroups.
  
    @return The connection probability.
  */
  double getConnectionProbability() const;
private:
  // The type of Synapse to simulate
  std::string type_;

  // The NeuronGroup that affects these synapses
  NeuronGroup* presynaptic_group_;
  
  // The NeuronGroup that is affected by these synapses
  NeuronGroup* postsynaptic_group_;

  // The probability that a connection exists from the presynaptic group to
  // the postsynaptic one
  double connection_probability_;

  // Parameters used to instantiate these neurons
  Specification specification_;
};

/**
  A container for multiple groups of Synapses.
*/
class SynapseCluster : public SynapseGroup {
public:
  /**
    Constructor.

    @param name The name of this higher level group.
  */
  SynapseCluster(const std::string& name);

  /**
    Returns all the SynapseSpecifications contained within this group.

    @return A list of all SynapseSpecifications contained in this group.
  */
  virtual void getSpecifications(SynapseSpecificationList* specs);

  /**
    Adds a SynapseGroup as a subgroup for this group.

    @param subgroup The child subgroup to add.
  */
  virtual void addGroup(SynapseGroup* subgroup);

  /**
    Returns the direct subgroups that compose this SynapseGroup if any.

    @return A list of the direct subgroups that compose this SynapseGroup.
  */
  virtual SynapseGroupList getSubgroups() const;
private:
  // All direct descendants of this group
  SynapseGroupList subgroups_;
};

} // namespace spec

} // namespace slug
