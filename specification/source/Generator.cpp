#include "Generator.h"
#include <iostream>

namespace slug {

namespace spec {

std::string Generator::generateString(RNG* rng) {
  std::cerr << name() << " does not generate strings." << std::endl;
  std::terminate();
}

std::int64_t Generator::generateInt(RNG* rng) {
  std::cerr << name() << " does not generate integers." << std::endl;
  std::terminate();
}

double Generator::generateDouble(RNG* rng) {
  std::cerr << name() << " does not generate floats." << std::endl;
  std::terminate();
}

} // namespace spec

} // namespace slug
