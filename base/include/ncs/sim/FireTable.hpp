namespace ncs {

namespace sim {

template<DeviceType::Type MType>
FireTable<MType>::FireTable(size_t device_synaptic_vector_size,
                            unsigned int min_delay,
                            unsigned int max_delay)
  : device_synaptic_vector_size_(device_synaptic_vector_size),
    min_delay_(min_delay),
    max_delay_(max_delay) {
}

template<DeviceType::Type MType>
bool FireTable<MType>::init() {
}

template<DeviceType::Type MType>
Bit::Word* FireTable<MType>::getTable() {
}

template<DeviceType::Type MType>
Bit::Word* FireTable<MType>::getRow(unsigned int index) {
}

template<DeviceType::Type MType>
size_t FireTable<MType>::getNumberOfRows() const {
}

template<DeviceType::Type MType>
size_t FireTable<MType>::getWordsPerVector() const {
}

template<DeviceType::Type MType>
FireTable<MType>::~FireTable() {
}


} // namespace sim

} // namespace ncs
