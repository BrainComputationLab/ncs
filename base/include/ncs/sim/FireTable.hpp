namespace ncs {

namespace sim {

template<DeviceType::Type MType>
FireTable<MType>::FireTable(size_t device_synaptic_vector_size,
                            unsigned int min_delay,
                            unsigned int max_delay)
  : device_synaptic_vector_size_(device_synaptic_vector_size),
    min_delay_(min_delay),
    max_delay_(max_delay),
    table_(nullptr) {
}

template<DeviceType::Type MType>
bool FireTable<MType>::init() {
  if (max_delay_ < min_delay_) {
    std::cerr << "Min delay is greater than max delay." << std::endl;
    return false;
  }
  num_rows_ = max_delay_ - min_delay_ + 1;
  size_t words_per_vector = getWordsPerVector();
  size_t total_words = words_per_vector * num_rows_;
  return Memory<MType>::malloc(table_, total_words);
}

template<DeviceType::Type MType>
Bit::Word* FireTable<MType>::getTable() {
  return table_;
}

template<DeviceType::Type MType>
Bit::Word* FireTable<MType>::getRow(unsigned int index) {
  size_t words_per_vector = getWordsPerVector();
  return table_ + words_per_vector * index;
}

template<DeviceType::Type MType>
size_t FireTable<MType>::getNumberOfRows() const {
  return num_rows_;
}

template<DeviceType::Type MType>
size_t FireTable<MType>::getWordsPerVector() const {
  return Bit::num_words(device_synaptic_vector_size_);
}

template<DeviceType::Type MType>
FireTable<MType>::~FireTable() {
  if (table_) {
    Memory<MType>::free(table_);
  }
}

} // namespace sim

} // namespace ncs
