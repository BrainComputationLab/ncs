#pragma once

#include <ncs/spec/Generator.h>
#include <ncs/spec/GeometryGenerator.h>

namespace ncs {

namespace spec {

class BoxGenerator : public GeometryGenerator {
public:
  BoxGenerator(Generator* x,
               Generator* y,
               Generator* z);
  virtual Geometry generate(RNG* rng);
private:
  Generator* x_;
  Generator* y_;
  Generator* z_;
};

} // namespace spec

} // namespace ncs
