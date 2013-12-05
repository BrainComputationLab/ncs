#include <ncs/sim/MemoryExtractor.h>

namespace ncs {

namespace sim {

MemoryExtractor::~MemoryExtractor() {
}

template<>
bool CPUExtractor<Bit>::extract(const void* source, void* destination) {
  const Bit::Word* s = static_cast<const Bit::Word*>(source);
  Bit::Word* d = static_cast<Bit::Word*>(destination);
  auto num_words = Bit::num_words(indices_.size());
  for (size_t i = 0; i < num_words; ++i) {
    d[i] = 0u;
  }
  for (size_t i = 0; i < indices_.size(); ++i) {
    auto dest_word_index = Bit::word(i);
    auto dest_mask = Bit::mask(i);
    auto index = indices_[i];
    auto source_word_index = Bit::word(index);
    auto source_word = s[source_word_index];
    if (Bit::extract(source_word, index)) {
      d[dest_word_index] |= dest_mask;
    }
  }
  return true;
}

} // namespace sim

} // namespace ncs
