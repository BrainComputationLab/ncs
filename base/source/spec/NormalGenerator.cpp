#include <ncs/spec/NormalGenerator.h>
#include "ModelParameters.pb.h"

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

bool NormalDouble::makeProtobuf(com::Generator* gen) const {
  gen->set_distribution(com::Generator::Normal);
  gen->set_value_type(com::Generator::Double);
  gen->set_mean_normal_double(mean_);
  gen->set_stddev_normal_double(std_dev_);
  return true;
}

} // namespace spec

} // namespace ncs
