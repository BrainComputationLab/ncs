#pragma once

namespace ncs {

namespace sim {

struct Bit {
  static unsigned int bits_per_word;
  static unsigned int num_words(unsigned int num_elements);
  static unsigned int pad(unsigned int original);
  static unsigned int mask(unsigned int position);
  static bool extract(unsigned int word, unsigned int position);
};

} // namespace sim

} // namespace ncs
