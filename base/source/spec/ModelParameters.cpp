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

bool ModelParameters::get(ncs::spec::Generator*& target,
                          const std::string& parameter_name) {
  auto generator = getGenerator(parameter_name);
  target = generator;
  if (!generator) {
    std::cerr << getType() << " requires " << parameter_name << 
      " to be defined." << std::endl;
    return false;
  }
  return true;
}

} // namespace spec

} // namespace ncs
