#include <ncs/sim/Memory.h>

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
  num_rows_ = max_delay_ + 1;
  row_free_.resize(num_rows_);
  for (size_t i = 0; i < num_rows_; ++i) {
    row_free_[i] = true;
  }
  size_t words_per_vector = getWordsPerVector();
  size_t total_words = words_per_vector * num_rows_;
  bool result = Memory<MType>::malloc(table_, total_words);
  if (!result) {
    std::cerr << "Failed to allocate FireTable." << std::endl;
    return false;
  }
  result &= Memory<MType>::zero(table_, total_words);
  if (!result) {
    std::cerr << "Failed to zero FireTable." << std::endl;
    return false;
  }
  return result;
}

template<DeviceType::Type MType>
Bit::Word* FireTable<MType>::getTable() {
  return table_;
}

template<DeviceType::Type MType>
Bit::Word* FireTable<MType>::getRow(unsigned int index) {
  size_t words_per_vector = getWordsPerVector();
  return table_ + words_per_vector * (index % getNumberOfRows());
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
bool FireTable<MType>::lockRow(unsigned int index) {
  unsigned int array_index = index % getNumberOfRows();
  std::unique_lock<std::mutex> lock(row_lock_);
  while (!row_free_[array_index]) {
    row_freed_.wait(lock);
  }
  row_free_[array_index] = false;
  return true;
}

template<DeviceType::Type MType>
bool FireTable<MType>::releaseRow(unsigned int index) {
  std::unique_lock<std::mutex> lock(row_lock_);
  row_free_[index % getNumberOfRows()] = true;
  row_freed_.notify_all();
  return true;
}

template<DeviceType::Type MType>
unsigned int FireTable<MType>::getMinDelay() const {
  return min_delay_;
}

template<DeviceType::Type MType>
unsigned int FireTable<MType>::getMaxDelay() const {
  return max_delay_;
}

template<DeviceType::Type MType>
FireTable<MType>::~FireTable() {
  if (table_) {
    Memory<MType>::free(table_);
  }
}

} // namespace sim

} // namespace ncs
