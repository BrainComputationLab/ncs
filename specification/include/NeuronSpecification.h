#pragma once

#include <GeometryGenerator.h>
#include <Specification.h>

namespace slug {

namespace spec {

// Forward declarations and container typedefs
class NeuronSpecification;
typedef std::vector<NeuronSpecification*> NeuronSpecificationList;

class NeuronGroup;
typedef std::vector<NeuronGroup*> NeuronGroupList;

class NeuronCluster;
typedef std::vector<NeuronCluster*> NeuronClusterList;

/**
  Abstract base class for a group of neurons.
*/
class NeuronGroup {
public:
  /**
    Constructor.

    @param name The name of this group.
  */
  NeuronGroup(const std::string& name);

  /**
    Returns all the NeuronSpecifications contained within this group.
    
    @param specs The list to output all NeuronSpecifications to.
  */
  virtual void getSpecifications(NeuronSpecificationList* specs);

  /**
    Returns the direct subgroups that compose this NeuronGroup if any.

    @return A list of the direct subgroups that compose this NeuronGroup.
  */
  virtual NeuronGroupList getSubgroups();

  /**
    Returns the name of this group.

    @return The name of this group.
  */
  const std::string& getName() const;
private:
  // The name of this group.
  std::string name_;
};

/**
  The manifestation of a set of Neurons. Unlike a NeuronCluster, a
  NeuronSpecification actually instantiates Neurons rather than simply
  organizing them.
*/
class NeuronSpecification : public NeuronGroup {
public:
  /**
    Constructor.

    @param type The type of Neuron to instantiate.
    @param name The name of this set of Neurons.
    @param count The number of Neurons to instantiate.
    @param specification Parameters used to generate the Neuron.
    @param geometry_generator The generator used to position generated Neurons.
  */
  NeuronSpecification(const std::string& type,
                      const std::string& name,
                      unsigned int count,
                      const Specification& specification,
                      GeometryGenerator* geometry_generator);

  /**
    Returns all the NeuronSpecifications contained within this group.
    
    @param specs The list to output all NeuronSpecifications to.
  */
  virtual void getSpecifications(NeuronSpecificationList* specs);
private:
  // The type of Neuron to instantiate
  std::string type_;

  // The number of Neurons to instantiate
  unsigned int count_;

  // Parameters used to instantiate Neurons
  Specification specification_;

  // Generates positions for each Neuron
  GeometryGenerator* geometry_generator_;
};

/**
  Encapsulates a cluster of NeuronGroups.
*/
class NeuronCluster : public NeuronGroup {
public:
  /**
    Constructor.

    @param name The name of this higher level group.
  */
  NeuronCluster(const std::string& name);

  /**
    Adds a NeuronGroup as a subgroup for this group.

    @param subgroup The child subgroup to add.
  */
  virtual void addGroup(NeuronGroup* subgroup);

  /**
    Returns all the NeuronSpecifications contained within this group.
    
    @param specs The list to output all NeuronSpecifications to.
  */
  virtual void getSpecifications(NeuronSpecificationList* specs);

  /**
    Returns the direct subgroups that compose this NeuronGroup if any.

    @return A list of the direct subgroups that compose this NeuronGroup.
  */
  virtual NeuronGroupList getSubgroups();
private:
  // All direct descendants of this group
  NeuronGroupList subgroups_;
};

} // namespace spec

} // namespace slug
