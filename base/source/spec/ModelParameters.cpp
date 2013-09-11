#include <iostream>

#include <ncs/spec/ModelParameters.h>

namespace ncs {

namespace spec {

ModelParameters::ModelParameters(const std::string& type,
                                 std::map<std::string, Generator*> parameters)
  : type_(type),
    parameters_(parameters) {
}

const std::string& ModelParameters::getType() const {
  return type_;
}

Generator* ModelParameters::getGenerator(const std::string& parameter_name) {
  auto search_result = parameters_.find(parameter_name);
  if (search_result == parameters_.end()) {
    std::cerr << "Could not find parameter " << parameter_name << std::endl;
    return nullptr;
  }
  return search_result->second;
}

} // namespace spec

} // namespace ncs
