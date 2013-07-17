#include "SynapseSpecification.h"

namespace slug {

namespace spec {

SynapseGroup::SynapseGroup(const std::string& name)
  : name_(name) {
}

void SynapseGroup::getSpecifications(SynapseSpecificationList* specs) {
}

SynapseGroupList SynapseGroup::getSubgroups() const {
  SynapseGroupList empty;
  return empty;
}

SynapseSpecification::SynapseSpecification(const std::string& type, 
                                           const std::string& name,
                                           NeuronGroup* presynaptic_group,
                                           NeuronGroup* postsynaptic_group,
                                           double connection_probability,
                                           const Specification& specification)
  : SynapseGroup(name),
    type_(type),
    presynaptic_group_(presynaptic_group),
    postsynaptic_group_(postsynaptic_group),
    connection_probability_(connection_probability),
    specification_(specification) {
}

void SynapseSpecification::getSpecifications(SynapseSpecificationList* specs) {
  specs->push_back(this);
}

NeuronGroup* SynapseSpecification::getPresynapticGroup() const {
  return presynaptic_group_;
}

NeuronGroup* SynapseSpecification::getPostsynapticGroup() const {
  return postsynaptic_group_;
}

double SynapseSpecification::getConnectionProbability() const {
  return connection_probability_;
}

SynapseCluster::SynapseCluster(const std::string& name)
  : SynapseGroup(name) {
}

void SynapseCluster::addGroup(SynapseGroup* subgroup) {
  subgroups_.push_back(subgroup);
}

void SynapseCluster::getSpecifications(SynapseSpecificationList* specs) {
  for (auto& subgroup : subgroups_) {
    subgroup->getSpecifications(specs);
  }
}

SynapseGroupList SynapseCluster::getSubgroups() const {
  return subgroups_;
}



} // namespace spec

} // namespace slug
