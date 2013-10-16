#pragma once
#ifndef SWIG
#include <string>
#include <vector>
#endif // SWIG

namespace ncs {

namespace spec {

class Report {
public:
  enum Target {
    Neuron = 0,
    Synapse = 1,
    Input = 2
  };
  Report(const std::vector<std::string>& aliases,
         Target target,
         const std::string& report_name);
  const std::vector<std::string>& getAliases() const;
  Target getTarget() const;
  const std::string& getReportName() const;
  ~Report();
private:
  std::vector<std::string> aliases_;
  Target target_;
  std::string report_name_;
};

} // namespace spec

} // namespace ncs
