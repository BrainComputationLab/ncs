#include <ncs/sim/ClusterDescription.h>

namespace ncs {

namespace sim {

ClusterDescription::
ClusterDescription(const std::vector<MachineDescription*>& machines) {
}

const std::vector<MachineDescription*>&
ClusterDescription::getMachines() const {
  return machines_;
}

double ClusterDescription::estimateTotalPower() const {
  double total_compute_power = 0.0;
  for (auto machine : getMachines()) {
    for (auto device : machine->getDevices()) {
      total_compute_power += device->getPower();
    }
  }
  return total_compute_power;
}

unsigned int ClusterDescription::getThisMachineIndex() const {
  return this_machine_index_;
}

MachineDescription* ClusterDescription::getThisMachine() const {
  return machines_[getThisMachineIndex()];
}

ClusterDescription*
ClusterDescription::getThisCluster(Communicator* communicator,
                                   unsigned int enabled_device_types) {
  MachineDescription* thisMachine =
    MachineDescription::getThisMachine(enabled_device_types);
  std::vector<MachineDescription*> machines;

  unsigned int this_machine_index = 0;
  // Get the device power of all other devices in the cluster
  std::vector<DeviceDescription*> devices;
  for (int i = 0; i < communicator->getNumProcesses(); ++i) {
    if (communicator->getRank() == i) {
      unsigned int num_devices = thisMachine->getDevices().size();
      if (!communicator->bcast(num_devices, i)) {
        std::cerr << "Failed to broadcast device count." << std::endl;
        return nullptr;
      }
      if (0 == num_devices) {
        continue;
      }
      std::vector<double> powers;
      for (auto& device : thisMachine->getDevices()) {
        powers.push_back(device->getPower());
      }
      if (!communicator->bcast(powers.data(), num_devices, i)) {
        std::cerr << "Failed to broadcast device powers." << std::endl;
        return nullptr;
      }
      this_machine_index = machines.size();
      machines.push_back(thisMachine);
    } else {
      unsigned int num_devices = 0;
      if (!communicator->bcast(num_devices, i)) {
        std::cerr << "Failed to recv device count." << std::endl;
        return nullptr;
      }
      if (0 == num_devices) {
        continue;
      }
      double* powers = new double[num_devices];
      if (!communicator->bcast(powers, num_devices, i)) {
        std::cerr << "Failed to recv device powers." << std::endl;
        return nullptr;
      }
      for (int j = 0; j < num_devices; ++j) {
        devices.push_back(new DeviceDescription(false,
                                                powers[j],
                                                DeviceType::UNKNOWN));
      }
      delete [] powers;
    }
  }
  ClusterDescription* cluster = new ClusterDescription(machines);
  cluster->this_machine_index_ = this_machine_index;
  return cluster;
}

} // namespace sim

} // namespace ncs
