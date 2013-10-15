#include <ncs/sim/ReportManager.h>

namespace ncs {

namespace sim {

ReportManager::ReportManager() {
}

bool ReportManager::addReportDataspace(const std::string& report_name,
                                       Dataspace::Space dataspace) {
  if (dataspace_by_report_name_.count(report_name) != 0) {
    if (dataspace_by_report_name_[report_name] != dataspace) {
      std::cerr << "Multiple dataspaces specified for report type " <<
        report_name << std::endl;
      return false;
    }
  }
  dataspace_by_report_name_[report_name] = dataspace;
  return true;
}

Dataspace::Space ReportManager::
getReportDataspace(const std::string& report_name) const {
  auto result = dataspace_by_report_name_.find(report_name);
  if (result == dataspace_by_report_name_.end()) {
    return Dataspace::Unknown;
  }
  return result->second;
}

bool ReportManager::addReportSource(const std::string& report_name,
                                    int machine_location,
                                    int device_location,
                                    int plugin_location,
                                    Publisher* source_publisher) {
  Location location(machine_location, device_location, plugin_location);
  auto& publisher_by_name = publisher_by_name_by_location_[location];
  if (publisher_by_name.count(report_name) != 0) {
    std::cerr << "A publisher for report " << report_name <<
      " was already registered." << std::endl;
    std::cerr << "machine_location: " << machine_location << std::endl;
    std::cerr << "device_location: " << device_location << std::endl;
    std::cerr << "plugin_location: " << plugin_location << std::endl;
    return false;
  }
  publisher_by_name[report_name] = source_publisher;
  return true;
}

Publisher* ReportManager::getReportSource(const std::string& report_name,
                                          int machine_location,
                                          int device_location,
                                          int plugin_location) const {
  Location location(machine_location, device_location, plugin_location);
  auto location_result = publisher_by_name_by_location_.find(location);
  if (location_result == publisher_by_name_by_location_.end()) {
    std::cerr << "No sources registered for location." << std::endl;
    std::cerr << "machine_location: " << machine_location << std::endl;
    std::cerr << "device_location: " << device_location << std::endl;
    std::cerr << "plugin_location: " << plugin_location << std::endl;
    return nullptr;
  }
  auto& publisher_by_name = location_result->second;
  auto name_result = publisher_by_name.find(report_name);
  if (name_result == publisher_by_name.end()) {
    std::cerr << "No report source for report " << report_name <<
      " found." << std::endl;
    std::cerr << "machine_location: " << machine_location << std::endl;
    std::cerr << "device_location: " << device_location << std::endl;
    std::cerr << "plugin_location: " << plugin_location << std::endl;
    return nullptr;
  }
  return name_result->second;
}

ReportManager::Location::Location(int m, int d, int p)
  : machine(m),
    device(d),
    plugin(p) {
}

bool ReportManager::Location::operator==(const Location& r) const {
  return machine == r.machine &&
    device == r.device &&
    plugin == r.plugin;
}

bool ReportManager::Location::operator<(const Location& r) const {
  if (machine < r.machine) {
    return true;
  }
  if (machine > r.machine) {
    return false;
  }
  // Machines are equal
  if (device < r.device) {
    return true;
  }
  if (device > r.device) {
    return false;
  }
  // Devices are equal
  if (plugin < r.plugin) {
    return true;
  }
  return false;
}

ReportManager::~ReportManager() {
}

ReportManagers::ReportManagers() 
  : neuron_manager_(new ReportManager()),
    synapse_manager_(new ReportManager()) {
}

ReportManager* ReportManagers::getNeuronManager() {
  return neuron_manager_;
}

ReportManager* ReportManagers::getSynapseManager() {
  return synapse_manager_;
}

ReportManagers::~ReportManagers() {
  delete neuron_manager_;
  delete synapse_manager_;
}

} // namespace sim

} // namespace ncs
