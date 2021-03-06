#include <ncs/spec/Report.h>
#include "ModelParameters.pb.h"

namespace ncs {

namespace spec {

Report::Report(const std::vector<std::string>& aliases,
       Target target,
       const std::string& report_name,
       float percentage,
       double start_time,
       double end_time)
  : aliases_(aliases),
    target_(target),
    report_name_(report_name),
    percentage_(percentage),
    start_time_(start_time),
    end_time_(end_time) {
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

float Report::getPercentage() const {
  return percentage_;
}

float Report::getStartTime() const {
  return start_time_;
}

float Report::getEndTime() const {
  return end_time_;
}

Report::~Report() {
}

bool Report::toProtobuf(com::Report* report) const {
  for (auto alias : aliases_) {
    report->add_alias(alias);
  }
  switch(target_) {
    case Neuron:
      report->set_target(com::Report::Neuron);
      break;
    case Synapse:
      report->set_target(com::Report::Synapse);
      break;
    case Input:
      report->set_target(com::Report::Input);
      break;
    default:
      report->set_target(com::Report::Unknown);
  }
  report->set_report_name(report_name_);
  report->set_percentage(percentage_);
  report->set_start_time(start_time_);
  report->set_end_time(end_time_);
  return true;
}

Report* Report::fromProtobuf(com::Report* report) {
  std::vector<std::string> aliases;
  for (int i = 0; i < report->alias_size(); ++i) {
    aliases.push_back(report->alias(i));
  }
  Target target;
  switch(report->target()) {
    case com::Report::Neuron:
      target = Neuron;
      break;
    case com::Report::Synapse:
      target = Synapse;
      break;
    case com::Report::Input:
      target = Input;
      break;
    default:
      target = Unknown;
      break;
  }
  return new Report(aliases,
                    target,
                    report->report_name(),
                    report->percentage(),
                    report->start_time(),
                    report->end_time());
}

} // namespace spec

} // namespace ncs
