#include "ExactGenerator.h"

namespace slug {

namespace spec {

ExactInteger::ExactInteger(std::int64_t value)
	: value_(value) {
}

std::int64_t ExactInteger::generateInt(RNG* rng) {
	return value_;
}

double ExactInteger::generateDouble(RNG* rng) {
	return value_;
}

const std::string& ExactInteger::name() const {
	static std::string n = "ExactInteger";
	return n;
}

ExactDouble::ExactDouble(double value)
	: value_(value) {
}

double ExactDouble::generateDouble(RNG* rng) {
	return value_;
}

const std::string& ExactDouble::name() const {
	static std::string n = "ExactDouble";
	return n;
}

ExactString::ExactString(const std::string& value)
	: value_(value) {
}

std::string ExactString::generateString(RNG* rng) {
	return value_;
}

const std::string& ExactString::name() const {
	static std::string n = "ExactString";
	return n;
}

} // namespace spec

} // namespace slug
