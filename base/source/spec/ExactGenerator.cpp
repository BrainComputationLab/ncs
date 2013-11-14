#include <ncs/spec/ExactGenerator.h>
#include "ModelParameters.pb.h"

namespace ncs {

namespace spec {

ExactInteger::ExactInteger(long value)
  : value_(value) {
}

long ExactInteger::generateInt(RNG* rng) {
  return value_;
}

double ExactInteger::generateDouble(RNG* rng) {
  return value_;
}

const std::string& ExactInteger::name() const {
  static std::string n = "ExactInteger";
  return n;
}

bool ExactInteger::makeProtobuf(com::Generator* gen) const {
  gen->set_distribution(com::Generator::Exact);
  gen->set_value_type(com::Generator::Integer);
  gen->set_exact_integer(value_);
  return true;
}

ExactDouble::ExactDouble(double value)
  : value_(value) {
}

double ExactDouble::generateDouble(RNG* rng) {
  return value_;
}

const std::string& ExactDouble::name() const {
  static std::string n = "ExactDouble";
  return n;
}

bool ExactDouble::makeProtobuf(com::Generator* gen) const {
  gen->set_distribution(com::Generator::Exact);
  gen->set_value_type(com::Generator::Double);
  gen->set_exact_double(value_);
  return true;
}

ExactString::ExactString(const std::string& value)
  : value_(value) {
}

std::string ExactString::generateString(RNG* rng) {
  return value_;
}

const std::string& ExactString::name() const {
  static std::string n = "ExactString";
  return n;
}

bool ExactString::makeProtobuf(com::Generator* gen) const {
  gen->set_distribution(com::Generator::Exact);
  gen->set_value_type(com::Generator::String);
  gen->set_exact_string(value_);
  return true;
}

ExactList::ExactList(const std::vector<Generator*>& value)
  : value_(value) {
}

std::vector<Generator*> ExactList::generateList(RNG* rng) {
  return value_;
}

const std::string& ExactList::name() const {
  static std::string n = "ExactList";
  return n;
}

bool ExactList::makeProtobuf(com::Generator* gen) const {
  gen->set_distribution(com::Generator::Exact);
  gen->set_value_type(com::Generator::List);
  for (auto value : value_) {
    value->makeProtobuf(gen->add_exact_list());
  }
  return true;
}

ExactParameters::ExactParameters(ModelParameters* value)
  : value_(value) {
}

ModelParameters* ExactParameters::generateParameters(RNG* rng) {
  return value_;
}

const std::string& ExactParameters::name() const {
  static std::string n = "ExactParameters";
  return n;
}

bool ExactParameters::makeProtobuf(com::Generator* gen) const {
  gen->set_distribution(com::Generator::Exact);
  gen->set_value_type(com::Generator::Parameters);
  value_->makeProtobuf(gen->mutable_exact_parameters());
  return true;
}

ExactParameters::~ExactParameters() {
  if (value_) {
    delete value_;
  }
}

} // namespace spec

} // namespace ncs
