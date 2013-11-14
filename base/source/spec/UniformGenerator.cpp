#include <ncs/spec/UniformGenerator.h>
#include "ModelParameters.pb.h"

namespace ncs {

namespace spec {

UniformInteger::UniformInteger(long min, long max)
  : min_value_(min),
    max_value_(max) {
}

long UniformInteger::generateInt(RNG* rng) {
  std::uniform_int_distribution<long> distribution(min_value_, max_value_);
  return distribution(*rng);
}

const std::string& UniformInteger::name() const {
  static std::string n = "UniformInteger";
  return n;
}

bool UniformInteger::makeProtobuf(com::Generator* gen) const {
  gen->set_distribution(com::Generator::Uniform);
  gen->set_value_type(com::Generator::Integer);
  gen->set_min_uniform_integer(min_value_);
  gen->set_max_uniform_integer(max_value_);
  return true;
}

UniformDouble::UniformDouble(double min, double max)
  : min_value_(min),
    max_value_(max) {
}

double UniformDouble::generateDouble(RNG* rng) {
  std::uniform_real_distribution<double> distribution(min_value_, max_value_);
  return distribution(*rng);
}

const std::string& UniformDouble::name() const {
  static std::string n = "UniformDouble";
  return n;
}

bool UniformDouble::makeProtobuf(com::Generator* gen) const {
  gen->set_distribution(com::Generator::Uniform);
  gen->set_value_type(com::Generator::Double);
  gen->set_min_uniform_double(min_value_);
  gen->set_max_uniform_double(max_value_);
  return true;
}

} // namespace spec

} // namespace ncs
