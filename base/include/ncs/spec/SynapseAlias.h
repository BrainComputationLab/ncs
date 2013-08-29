#pragma once
#ifndef SWIG
#include <vector>
#endif // SWIG

#include <ncs/spec/SynapseGroup.h>

namespace ncs {

namespace spec {

/**
  A construct that represents a set of SynapseGroups.
*/
class SynapseAlias {
public:
  /**
    Constructor.

    @param groups The groups that this alias represents.
  */
  SynapseAlias(const std::vector<SynapseGroup*>& groups);

  /**
    Returns a list of all groups this alias represents.
    
    @return A list of all groups this alias represents.
  */
  const std::vector<SynapseGroup*>& getGroups() const;
private:
  /// All the groups this alias represents.
  std::vector<SynapseGroup*> groups_;
};

} // namespace spec

}
