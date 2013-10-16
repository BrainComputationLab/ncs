#include <ncs/spec/Report.h>

namespace ncs {

namespace spec {

Report::Report(const std::vector<std::string>& aliases,
       Target target,
       const std::string& report_name)
  : aliases_(aliases),
    target_(target),
    report_name_(report_name) {
}

const std::vector<std::string>& Report::getAliases() const {
  return aliases_;
}

Report::Target Report::getTarget() const {
  return target_;
}

const std::string& Report::getReportName() const {
  return report_name_;
}

Report::~Report() {
}

} // namespace spec

} // namespace ncs
