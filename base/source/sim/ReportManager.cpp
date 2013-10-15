#include <ncs/sim/ReportManager.h>

namespace ncs {

namespace sim {

ReportManager::ReportManager() {
}

bool ReportManager::addReportDataspace(const std::string& report_name,
                                       Dataspace dataspace) {
}

Dataspace ReportManager::
getReportDataspace(const std::string& report_name) const {
}

bool ReportManager::addReportSource(const std::string& report_name,
                                    int machine_location,
                                    int device_location,
                                    int plugin_location,
                                    Publisher* source_publisher) {
}

Publisher* ReportManager::getReportSource(const std::string& report_name,
                                          int machine_location,
                                          int device_location,
                                          int plugin_location) const {
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
