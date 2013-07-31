#include <ncs/spec/SynapseAlias.h>

namespace ncs {

namespace spec {

SynapseAlias::SynapseAlias(const std::vector<SynapseGroup*>& groups)
  : groups_(groups) {
}

const std::vector<SynapseGroup*>& SynapseAlias::getGroups() const {
  return groups_;
}

} // namespace spec

} // namespace ncs

