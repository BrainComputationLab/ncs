#pragma once
#include <ncs/sim/DataBuffer.h>
#include <ncs/sim/DataDescription.h>
#include <ncs/sim/Location.h>

namespace ncs {

namespace sim {

class ReportManager {
public:
  ReportManager();
  bool addDescription(const std::string& report_name,
                      const DataDescription& description);
  const DataDescription* getDescription(const std::string& report_name) const;
  bool addSource(const std::string& report_name,
                 int machine_location,
                 int device_location,
                 int plugin_location,
                 Publisher* source_publisher);
  Publisher* getSource(const std::string& report_name,
                       int machine_location,
                       int device_location,
                       int plugin_location) const;
  ~ReportManager();
private:
  std::map<std::string, DataDescription*> description_by_report_name_;
  typedef std::map<std::string, Publisher*> PublisherByReportName;
  std::map<Location, PublisherByReportName> publisher_by_name_by_location_; 
};

class ReportManagers {
public:
  ReportManagers();
  ReportManager* getNeuronManager();
  ReportManager* getSynapseManager();
  ~ReportManagers();
private:
  ReportManager* neuron_manager_;
  ReportManager* synapse_manager_;
};

} // namespace sim

} // namespace ncs
