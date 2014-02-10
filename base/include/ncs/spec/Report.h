#pragma once
#ifndef SWIG
#include <string>
#include <vector>
#endif // SWIG

namespace ncs {

namespace com {
  struct Report;
}

namespace spec {

class Report {
public:
  enum Target {
    Neuron = 0,
    Synapse = 1,
    Input = 2,
    Unknown = 3
  };
  Report(const std::vector<std::string>& aliases,
         Target target,
         const std::string& report_name,
         float percentage,
         double start_time,
         double end_time);
  const std::vector<std::string>& getAliases() const;
  Target getTarget() const;
  const std::string& getReportName() const;
  float getPercentage() const;
  float getStartTime() const;
  float getEndTime() const;
  ~Report();
  bool toProtobuf(com::Report* report) const;
  static Report* fromProtobuf(com::Report* report);
private:
  std::vector<std::string> aliases_;
  Target target_;
  std::string report_name_;
  float percentage_;
  double start_time_;
  double end_time_;
};

} // namespace spec

} // namespace ncs
