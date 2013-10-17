namespace ncs {

namespace sim {

template<typename T>
CPUExtractor<T>::CPUExtractor(const std::vector<unsigned int>& indices)
  : indices_(indices) {
}

template<typename T>
bool CPUExtractor<T>::extract(const void* source, void* destination) {
  const T* s = static_cast<const T*>(source);
  T* d = static_cast<T*>(destination);
  for (size_t i = 0; i < indices_.size(); ++i) {
    unsigned int index = indices_[i];
    d[i] = s[index];
  }
  return true;
}

template<typename T>
CPUExtractor<T>::~CPUExtractor() {
}

} // namespace sim

} // namespace ncs
