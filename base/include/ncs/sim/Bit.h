#pragma once

namespace ncs {

namespace sim {

struct Bit {
  typedef unsigned int Word;
  static unsigned int bits_per_word;
  static unsigned int mod_mask;
  static unsigned int word_shifts;
  static unsigned int num_words(unsigned int num_elements);
  static unsigned int word(unsigned int position);
  static unsigned int pad(unsigned int original);
  static unsigned int mask(unsigned int position);
  static bool extract(Word word, unsigned int position);
};

} // namespace sim

} // namespace ncs
