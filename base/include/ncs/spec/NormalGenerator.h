#pragma once

#ifndef SWIG
#include <cstdint>
#include <random>
#include <string>
#endif // SWIG

#include <ncs/spec/Generator.h>

namespace ncs {

namespace spec {

class NormalDouble : public Generator {
public:
  NormalDouble(double mean, double std_dev);
  virtual double generateDouble(RNG* rng);
  virtual const std::string& name() const;
  virtual bool makeProtobuf(com::Generator* gen) const;
private:
  double mean_;
  double std_dev_;
};

}

}

