#pragma once

#include <map>

#include <ncs/sim/ClusterDescription.h>
#include <ncs/spec/ModelSpecification.h>
#include <ncs/spec/NeuronGroup.h>

namespace ncs {

namespace sim {

class Distributor {
public:
  bool distribute(spec::ModelSpecification* spec, ClusterDescription* cluster);
private:
};

} // namespace sim

} // namespace ncs
