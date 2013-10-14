namespace ncs {

namespace sim {

template<typename T>
AtomicWriter<T>::AtomicWriter() {
}

template<typename T>
void AtomicWriter<T>::write(T* location, const T& value) {
  locations_.push_back(location);
  values_.push_back(value);
}

template<typename T>
void AtomicWriter<T>::commit(std::function<void(T*, const T&)> op) {
  for (std::size_t i = 0; i < locations_.size(); ++i) {
    op(locations_[i], values_[i]);
  }
}

template<typename T>
void AtomicWriter<T>::Add(T* location, const T& value) {
  *location += value;
}

template<typename T>
void AtomicWriter<T>::Or(T* location, const T& value) {
  *location |= value;
}

template<typename T>
void AtomicWriter<T>::Set(T* location, const T& value) {
  *location = value;
}

} // namespace sim

} // namespace ncs
