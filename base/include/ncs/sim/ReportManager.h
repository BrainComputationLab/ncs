#pragma once
#include <ncs/sim/DataBuffer.h>
#include <ncs/sim/Dataspace.h>

namespace ncs {

namespace sim {

enum class ReportTarget {
  Neuron,
  Synapse
};

class ReportManager {
public:
  ReportManager();
  bool addReportDataspace(const std::string& report_name,
                          Dataspace dataspace);
  Dataspace getReportDataspace(const std::string& report_name) const;
  bool addReportSource(const std::string& report_name,
                       int machine_location,
                       int device_location,
                       int plugin_location,
                       Publisher* source_publisher);
  Publisher* getReportSource(const std::string& report_name,
                             int machine_location,
                             int device_location,
                             int plugin_location) const;
  ~ReportManager();
private:
  std::map<std::string, Dataspace> report_name_to_dataspace_;
  struct Location {
    Location(int m, int d, int p);
    bool operator==(const Location& r) const;
    bool operator<(const Location& r) const;
    int machine;
    int device;
    int plugin;
  };
  std::map<Location, Publisher*> location_to_publisher_;
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
