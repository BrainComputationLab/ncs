#include "NeuronSpecification.h"

namespace slug {

namespace spec {

NeuronGroup::NeuronGroup(const std::string& name)
	: name_(name) {
}

void NeuronGroup::getSpecifications(NeuronSpecificationList* specs) {
}

NeuronGroupList NeuronGroup::getSubgroups() {
	NeuronGroupList empty;
	return empty;
}

const std::string& NeuronGroup::getName() const {
	return name_;
}

NeuronSpecification::NeuronSpecification(const std::string& type,
                                         const std::string& name,
                                         unsigned int count,
                                         const Specification& specification,
                                         GeometryGenerator* geometry_generator)
  : NeuronGroup(name),
	  type_(type),
    count_(count),
    specification_(specification),
    geometry_generator_(geometry_generator) {
}

void NeuronSpecification::
getSpecifications(NeuronSpecificationList* specs) {
	specs->push_back(this);
}

NeuronCluster::NeuronCluster(const std::string& name)
	: NeuronGroup(name) {
}

void NeuronCluster::getSpecifications(NeuronSpecificationList* specs) {
	for (auto& subgroup : subgroups_) {
		subgroup->getSpecifications(specs);
	}
}

void NeuronCluster::addGroup(NeuronGroup* subgroup) {
	subgroups_.push_back(subgroup);
}

NeuronGroupList NeuronCluster::getSubgroups() {
	return subgroups_;
}

} // namespace spec

} // namespace slug
