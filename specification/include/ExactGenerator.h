#pragma once

#ifndef SWIG
#include <string>
#include <random>
#include <cstdint>
#endif // SWIG

#include "Generator.h"

namespace slug {

namespace spec {

/**
  Always returns the same exact integer value. Can also output this value
  as a double.
*/
class ExactInteger : public Generator {
public:
  /**
    Constructor.

    @param value The integer value to always generate.
  */
  ExactInteger(std::int64_t value);

  /**
    Generates an integer.

    @param rng A random number generator.
    @return A 64-bit integer.
  */
  virtual std::int64_t generateInt(RNG* rng);

  /**
    Generates a floating point value.

    @param rng A random number generator.
    @return A 64-bit floating point value.
  */
  virtual double generateDouble(RNG* rng);

  /**
    Returns the name of the generator. Useful for printing error messages.

    @return The name of this specific generator type.
  */
  virtual const std::string& name() const;
private:
  // The integer value to always generate.
  std::int64_t value_;
};

/**
  Always returns the same exact floating point value. Unlike ExactInteger, to
  prevent loss of precision, this value cannot be outputted as an integer.
*/
class ExactDouble : public Generator {
public:
  /**
    Constructor.

    @param value The floating point value to always generate.
  */
  ExactDouble(double value);

  /**
    Generates a floating point value.

    @param rng A random number generator.
    @return A 64-bit floating point value.
  */
  virtual double generateDouble(RNG* rng);

  /**
    Returns the name of the generator. Useful for printing error messages.

    @return The name of this specific generator type.
  */
  virtual const std::string& name() const;
private:
  // The floating point value to always generate.
  double value_;
};

/**
  Always returns the exact same string.
*/
class ExactString : public Generator {
public:
  /**
    Constructor.

    @param value The string to always generate.
  */
  ExactString(const std::string& value);

  /**
    Generates a string.

    @param rng A random number generator.
    @return A string.
  */
  virtual std::string generateString(RNG* rng);

  /**
    Returns the name of the generator. Useful for printing error messages.

    @return The name of this specific generator type.
  */
  virtual const std::string& name() const;
private:
  // The string to always generate.
  std::string value_;
};

} // namespace spec

} // namespace slug
