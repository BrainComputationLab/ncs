#pragma once

#include <thread>

#include <ncs/sim/Bit.h>
#include <ncs/sim/Constants.h>
#include <ncs/sim/DeviceNeuronStateBuffer.h>
#include <ncs/sim/GlobalNeuronStateBuffer.h>
#include <ncs/sim/GlobalFireVectorBuffer.h>
#include <ncs/sim/Signal.h>
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

class VectorExchangeBuffer : public DataBuffer {
public:
  VectorExchangeBuffer(size_t global_vector_size);
  bool init();
  Bit::Word* getData();
  ~VectorExchangeBuffer();
private:
  Bit::Word* data_vector_;
  size_t global_vector_size_;
};

class VectorExchangeController
  : public SpecificPublisher<VectorExchangeBuffer> {
public:
  VectorExchangeController();
  bool init(size_t global_vector_size,
            size_t num_buffers);
  bool start();
  ~VectorExchangeController();
private:
  std::thread thread_;
};

#if 0
template<DeviceType::Type MType>
class DeviceVectorExtractor : public SpecificPublisher<Signal> {
public:
  typedef DeviceNeuronStateBuffer<MType> SourceBuffer;
  typedef SpecificPublisher<SourceBuffer> SourcePublisher;
  typedef SpecificPublisher<VectorExchangeBuffer> DestinationPublisher;
  DeviceVectorExtractor();
  bool init(size_t global_word_offset,
            size_t num_buffers,
            SourcePublisher* source_publisher,
            DestinationPublisher* destination_publisher);
  bool start();
  ~DeviceVectorExtractor();
private:
  size_t global_word_offset_;
  typename SourcePublisher::Subscription* source_subscription_;
  typename DestinationPublisher::Subscription* destination_subscription_;
  std::thread thread_;
};
#endif

class GlobalVectorPublisher 
  : public SpecificPublisher<GlobalFireVectorBuffer> {
public:
  typedef SpecificPublisher<Signal> DependentPublisher;
  GlobalVectorPublisher();
  bool init(size_t global_vector_size,
            size_t num_buffers,
            const std::vector<DependentPublisher*>& dependent_publishers,
            SpecificPublisher<VectorExchangeBuffer>* source_publisher);
  bool start();
  ~GlobalVectorPublisher();
private:
  std::vector<DependentPublisher::Subscription*> dependent_subscriptions_;
  SpecificPublisher<VectorExchangeBuffer>::Subscription* source_subscription_;
  std::thread thread_;
};

} // namespace sim

} // namespace ncs

#include <ncs/sim/VectorExchanger.hpp>
