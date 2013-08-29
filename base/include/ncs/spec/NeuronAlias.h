#pragma once
#ifndef SWIG
#include <vector>
#endif // SWIG

#include <ncs/spec/NeuronGroup.h>

namespace ncs {

namespace spec {

/**
  A construct that represents a set of NeuronGroups.
*/
class NeuronAlias {
public:
  /**
    Constructor.

    @param groups The groups that this alias represents.
  */
  NeuronAlias(const std::vector<NeuronGroup*>& groups);

  /**
    Returns a list of all groups this alias represents.
    
    @return A list of all groups this alias represents.
  */
  const std::vector<NeuronGroup*>& getGroups() const;
private:
  /// All the groups this alias represents.
  std::vector<NeuronGroup*> groups_;
};

} // namespace spec

} //namespace ncs
