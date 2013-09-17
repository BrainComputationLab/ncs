namespace ncs {

namespace sim {

template<DeviceType::Type MType>
SynapticFireVectorBuffer<MType>::SynapticFireVectorBuffer(size_t num_words)
  : num_words_(num_words) {
}

template<DeviceType::Type MType>
bool SynapticFireVectorBuffer<MType>::setData(Bit::Word* data_row) {
  data_row_ = data_row;
  setPin_("synapse_fire", data_row_, MType);
  return true;
}

template<DeviceType::Type MType>
bool SynapticFireVectorBuffer<MType>::init() {
  return true;
}

template<DeviceType::Type MType>
size_t SynapticFireVectorBuffer<MType>::getNumberOfWords() const {
  return num_words_;
}

template<DeviceType::Type MType>
SynapticFireVectorBuffer<MType>::~SynapticFireVectorBuffer() {
}

} // namespace sim

} // namespace ncs
