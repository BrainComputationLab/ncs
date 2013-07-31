#include <ncs/spec/NeuronAlias.h>

namespace ncs {

namespace spec {

NeuronAlias::NeuronAlias(const std::vector<NeuronGroup*>& groups)
  : groups_(groups) {
}

const std::vector<NeuronGroup*>& NeuronAlias::getGroups() const {
  return groups_;
}

} // namespace spec

} // namespace ncs
