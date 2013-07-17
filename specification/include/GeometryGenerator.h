#pragma once

#include <Geometry.h>

#ifndef SWIG
#include <cstdint>
#include <random>
#include <string>
#endif // SWIG

namespace slug {

namespace spec {

typedef std::mt19937 RNG;

/**
  Abstract base class for a Generator that creates Geometry based on an RNG.
*/
class GeometryGenerator {
public:
  /**
    Generates geometry.
  */
  virtual Geometry generate(RNG& rng) = 0;
};

} // namespace spec

} // namespace slug
