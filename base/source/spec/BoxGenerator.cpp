#include <ncs/spec/BoxGenerator.h>

namespace ncs {

namespace spec {

BoxGenerator::BoxGenerator(Generator* x,
                           Generator* y,
                           Generator* z)
  : x_(x),
    y_(y),
    z_(z) {
}

Geometry BoxGenerator::generate(RNG* rng) {
  Geometry v;
  v.x = x_->generateDouble(rng);
  v.y = y_->generateDouble(rng);
  v.z = z_->generateDouble(rng);
  return v;
}

} // namespace spec

} // namespace ncs
