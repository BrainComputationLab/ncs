#include <ncs/spec/NormalGenerator.h>

namespace ncs {

namespace spec {

NormalDouble::NormalDouble(double mean, double std_dev)
  : mean_(mean),
    std_dev_(std_dev) {
}

double NormalDouble::generateDouble(RNG* rng) {
  std::normal_distribution<double> distribution(mean_, std_dev_);
  return distribution(*rng);
}

const std::string& NormalDouble::name() const {
  static std::string n = "NormalDouble";
  return n;
}

} // namespace spec

} // namespace ncs
