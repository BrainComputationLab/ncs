#include <ncs/spec/UniformGenerator.h>

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

} // namespace spec

} // namespace ncs
