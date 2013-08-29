#pragma once
#include <vector>

#include <ncs/sim/MachineDescription.h>
#include <ncs/sim/MPI.h>

namespace ncs {

namespace sim {

class ClusterDescription {
public:
  ClusterDescription(const std::vector<MachineDescription*>& machines);
  const std::vector<MachineDescription*>& getMachines() const;
  double estimateTotalPower() const;
  unsigned int getThisMachineIndex() const;
  MachineDescription* getThisMachine() const;
  static ClusterDescription*
    getThisCluster(Communicator* communicator,
                   unsigned int enabled_device_types);
private:
  std::vector<MachineDescription*> machines_;
  unsigned int this_machine_index_;
};

} // namespace sim

} // namespace ncs
