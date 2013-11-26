#pragma once

#include <thread>

#include <ncs/sim/Bit.h>
#include <ncs/sim/Constants.h>
#include <ncs/sim/DeviceNeuronStateBuffer.h>
#include <ncs/sim/GlobalNeuronStateBuffer.h>
#include <ncs/sim/GlobalFireVectorBuffer.h>
#include <ncs/sim/MPI.h>
#include <ncs/sim/Signal.h>
#include <ncs/sim/StepSignal.h>

namespace ncs {

namespace sim {

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

template<DeviceType::Type MType>
class DeviceVectorExtractor : public SpecificPublisher<Signal> {
public:
  typedef DeviceNeuronStateBuffer<MType> SourceBuffer;
  typedef SpecificPublisher<SourceBuffer> SourcePublisher;
  typedef SpecificPublisher<VectorExchangeBuffer> DestinationPublisher;
  DeviceVectorExtractor();
  bool init(size_t global_vector_offset,
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

class RemoteVectorExtractor : public SpecificPublisher<Signal> {
public:
  typedef SpecificPublisher<VectorExchangeBuffer> DestinationPublisher;
  RemoteVectorExtractor();
  bool init(size_t global_vector_offset,
            size_t machine_vector_size,
            Communicator* communicator, 
            int source_rank,
            DestinationPublisher* destination_publisher,
            size_t num_buffers);
  bool start();
  ~RemoteVectorExtractor();
private:
  size_t global_vector_offset_;
  size_t machine_vector_size_;
  Communicator* communicator_;
  int source_rank_;
  typename DestinationPublisher::Subscription* destination_subscription_;
  size_t num_buffers_;
  std::thread thread_;
};

class RemoteVectorPublisher {
public:
  typedef SpecificPublisher<VectorExchangeBuffer> SourcePublisher;
  typedef SpecificPublisher<Signal> DependentPublisher;
  RemoteVectorPublisher();
  bool init(size_t global_vector_offset,
            size_t machine_vector_size,
            Communicator* communicator,
            int destination_rank,
            SourcePublisher* source_publisher,
            const std::vector<DependentPublisher*>& dependent_publishers);
  bool start();
  ~RemoteVectorPublisher();
private:
  size_t global_vector_offset_;
  size_t machine_vector_size_;
  Communicator* communicator_;
  int destination_rank_;
  std::vector<DependentPublisher::Subscription*> dependent_subscriptions_;
  typename SourcePublisher::Subscription* source_subscription_;
  std::thread thread_;
};

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
  bool initialized_;
};

template<DeviceType::Type MType>
class GlobalVectorInjector
  : public SpecificPublisher<GlobalNeuronStateBuffer<MType>> {
public:
  GlobalVectorInjector();
  bool init(SpecificPublisher<GlobalFireVectorBuffer>* source_publisher,
            size_t global_neuron_vector_size,
            size_t num_buffers);
  bool start();
  ~GlobalVectorInjector();
private:
  SpecificPublisher<GlobalFireVectorBuffer>::Subscription*
    source_subscription_;
  size_t global_word_size_;
  std::thread thread_;
};

} // namespace sim

} // namespace ncs

#include <ncs/sim/VectorExchanger.hpp>
