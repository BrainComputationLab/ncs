#pragma once
#include <ncs/sim/Bit.h>
#include <ncs/sim/Constants.h>
#include <ncs/sim/DeviceNeuronStateBuffer.h>
#include <ncs/sim/GlobalNeuronStateBuffer.h>

namespace ncs {

namespace sim {

class DeviceVectorExtractorBase {
public:
  virtual bool pull(Bit::Word* dst) = 0;
  virtual ~DeviceVectorExtractorBase();
private:
};

template<DeviceType::Type MType>
class DeviceVectorExtractor : public DeviceVectorExtractorBase {
public:
  typedef SpecificPublisher<DeviceNeuronStateBuffer<MType>> StatePublisher;
  DeviceVectorExtractor();
  bool init(StatePublisher* publisher);
  virtual bool pull(Bit::Word* dst);
  virtual ~DeviceVectorExtractor();
private:
  typename StatePublisher::Subscription* state_subscription_;
};

class MachineVectorExchanger 
  : public SpecificPublisher<GlobalNeuronStateBuffer<DeviceType::CPU>> {
public:
  MachineVectorExchanger(size_t global_neuron_vector_size,
                         size_t num_buffers);
  bool init(const std::vector<DeviceVectorExtractorBase*>& device_extractors,
            const std::vector<size_t>& neuron_device_id_offsets);
private:
  size_t global_neuron_vector_size_;
  size_t num_buffers_;
  std::vector<DeviceVectorExtractorBase*> device_extractors_;
  std::vector<size_t> neuron_device_id_offsets_;
};

template<DeviceType::Type MType>
class GlobalVectorInjector
  : public SpecificPublisher<GlobalNeuronStateBuffer<MType>> {
public:
  GlobalVectorInjector(size_t global_neuron_vector_size,
                       size_t num_buffers);
  typedef SpecificPublisher<GlobalNeuronStateBuffer<DeviceType::CPU>>
    CPUGlobalPublisher;
  bool init(CPUGlobalPublisher* publisher);
  ~GlobalVectorInjector();
private:
  typename CPUGlobalPublisher::Subscription* subscription_;
  size_t global_neuron_vector_size_;
  size_t num_buffers_;
};

} // namespace sim

} // namespace ncs

#include <ncs/sim/VectorExchanger.hpp>
