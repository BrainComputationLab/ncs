#pragma once

#include <thread>

#include <ncs/sim/Bit.h>
#include <ncs/sim/Constants.h>
#include <ncs/sim/DeviceNeuronStateBuffer.h>
#include <ncs/sim/GlobalNeuronStateBuffer.h>
#include <ncs/sim/StepSignal.h>

namespace ncs {

namespace sim {

struct ExchangeStatus : public DataBuffer {
  bool valid;
};
typedef SpecificPublisher<ExchangeStatus> ExchangePublisher;
typedef std::vector<ExchangePublisher*> ExchangePublisherList;

class VectorExchanger
  : public SpecificPublisher<GlobalNeuronStateBuffer<DeviceType::CPU>> {
public:
  VectorExchanger();
  bool init(SpecificPublisher<StepSignal>* step_publisher,
            size_t global_neuron_vector_size,
            size_t num_buffers);
  bool start();
  virtual ~VectorExchanger();
private:
  size_t global_neuron_vector_size_;
  size_t num_buffers_;
  std::thread thread_;
  SpecificPublisher<StepSignal>::Subscription* step_subscription_;
};

template<DeviceType::Type MType>
class DeviceVectorExtractor : public SpecificPublisher<ExchangeStatus> {
public:
  typedef DeviceNeuronStateBuffer<MType> SourceBuffer;
  typedef SpecificPublisher<SourceBuffer> SourcePublisher;
  typedef GlobalNeuronStateBuffer<DeviceType::CPU> DestinationBuffer;
  typedef SpecificPublisher<DestinationBuffer> DestinationPublisher;
  DeviceVectorExtractor();
  bool init(size_t global_word_offset,
            size_t num_buffers);
  bool setDestinationPublisher(DestinationPublisher* publisher);
  bool setSourcePublisher(SourcePublisher* publisher);
  bool start();
  ~DeviceVectorExtractor();
private:
  typename SourcePublisher::Subscription* source_subscription_;
  typename DestinationPublisher::Subscription* destination_subscription_;
  size_t global_word_offset_;
  size_t num_buffers_;
  std::thread thread_;
};

class MachineVectorScatterer : public SpecificPublisher<ExchangeStatus> {
public:
  MachineVectorScatterer();
private:
};

class MachineVectorGatherer : public SpecificPublisher<ExchangeStatus> {
public:
private:
};

template<DeviceType::Type MType>
class GlobalVectorInjector
  : public SpecificPublisher<GlobalNeuronStateBuffer<MType>> {
public:
  typedef SpecificPublisher<GlobalNeuronStateBuffer<DeviceType::CPU>>
    CPUGlobalPublisher;
  GlobalVectorInjector();
  bool init(const ExchangePublisherList& dependent_publishers,
            CPUGlobalPublisher* buffer_publisher,
            size_t global_neuron_vector_size,
            size_t num_buffers);
  bool start();
  ~GlobalVectorInjector();
private:
  typedef typename ExchangePublisher::Subscription ExchangeSubscription;
  typename CPUGlobalPublisher::Subscription* source_subscription_;
  std::vector<ExchangeSubscription*> dependent_subscriptions_;
  size_t global_neuron_vector_size_;
  size_t num_buffers_;
  std::thread thread_;
};

} // namespace sim

} // namespace ncs

#include <ncs/sim/VectorExchanger.hpp>
