#pragma once

#ifndef SWIG
#include <cstdint>
#include <random>
#include <string>
#endif // SWIG

#include <ncs/spec/Generator.h>

namespace ncs {

namespace spec {

class UniformInteger : public Generator {
public:
  UniformInteger(long min, long max);
  virtual long generateInt(RNG* rng);
  virtual const std::string& name() const;
private:
  long min_value_;
  long max_value_;
};

class UniformDouble : public Generator {
public:
  UniformDouble(double min, double max);
  virtual double generateDouble(RNG* rng);
  virtual const std::string& name() const;
private:
  double min_value_;
  double max_value_;
};

}

}
