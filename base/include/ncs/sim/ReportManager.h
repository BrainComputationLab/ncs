#pragma once
#include <ncs/sim/DataBuffer.h>
#include <ncs/sim/DataType.h>
#include <ncs/sim/DataSpace.h>

namespace ncs {

namespace sim {

class DataDescription {
public:
  DataDescription(DataSpace::Space dataspace,
                  DataType::Type datatype);
  DataDescription(const DataDescription& source);
  DataSpace::Space getDataSpace() const;
  DataType::Type getDataType() const;
  bool operator==(const DataDescription& rhs) const;
  bool operator!=(const DataDescription& rhs) const;
private:
  DataSpace::Space dataspace_;
  DataType::Type datatype_;
};

class ReportManager {
public:
  ReportManager();
  bool addDescription(const std::string& report_name,
                      const DataDescription& description);
  DataDescription* getDescription(const std::string& report_name) const;
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
  struct Location {
    Location(int m, int d, int p);
    bool operator==(const Location& r) const;
    bool operator<(const Location& r) const;
    int machine;
    int device;
    int plugin;
  };
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
