#include <ncs/sim/Bit.h>

namespace ncs {

namespace sim {

unsigned int Bit::bits_per_word = sizeof(Bit::Word) * 8;
unsigned int Bit::mod_mask = 0x1F;
unsigned int Bit::word_shifts = 5;

unsigned int Bit::num_words(unsigned int num_elements) {
  return (num_elements + bits_per_word - 1) / bits_per_word;
}

unsigned int Bit::word(unsigned int position) {
  return position >> word_shifts;
}

unsigned int Bit::pad(unsigned int original) {
  return num_words(original) * bits_per_word;
}

unsigned int Bit::mask(unsigned int position) {
  position &= mod_mask;
  return 0x80000000 >> position;
}

bool Bit::extract(Word word, unsigned int position) {
  return word & mask(position);
}

} // namespace sim

} // namespace ncs
