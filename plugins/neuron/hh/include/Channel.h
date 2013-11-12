#pragma once
#include <ncs/sim/DataBuffer.h>

template<ncs::sim::DeviceType::Type MType>
class NeuronBuffer : public ncs::sim::DataBuffer {
public:
  NeuronBuffer();
  bool init(size_t num_neurons);
  void clear();
  float* getVoltage();
  ~NeuronBuffer();
private:
  float* voltage_;
};

template<ncs::sim::DeviceType::Type MType>
class ChannelCurrentBuffer : public ncs::sim::DataBuffer {
public:
  ChannelCurrentBuffer();
  bool init(size_t num_neurons);
  void clear();
  float* getCurrent();
  float* getReversalCurrent();
  ~ChannelCurrentBuffer();
private:
  float* current_;
  float* reversal_current_;
  size_t num_neurons_;
};

struct ChannelUpdateParameters {
  const float* voltage;
  float* current;
  float* reversal_current;
  float simulation_time;
  float time_step;
  std::mutex* write_lock;
};

template<ncs::sim::DeviceType::Type MType>
class ChannelSimulator {
public:
  ChannelSimulator();
  bool addChannel(void* instantiator,
                  unsigned int neuron_plugin_id,
                  int seed);
  bool initialize();
  virtual bool update(ChannelUpdateParameters* parameters) = 0;
  virtual ~ChannelSimulator();
protected:
  virtual bool init_() = 0;
  std::vector<void*> instantiators_;
  std::vector<unsigned int> cpu_neuron_plugin_ids_;
  std::vector<int> seeds_;
  unsigned int* neuron_plugin_ids_;
  size_t num_channels_;
private:
};

template<ncs::sim::DeviceType::Type MType>
class ChannelUpdater 
  : public ncs::sim::SpecificPublisher<ChannelCurrentBuffer<MType>> {
public:
  ChannelUpdater();
  typedef ncs::sim::SpecificPublisher<NeuronBuffer<MType>> NeuronPublisher;
  bool init(std::vector<ChannelSimulator<MType>*> simulators,
            NeuronPublisher* source_publisher,
            const ncs::spec::SimulationParameters* simulation_parameters,
            size_t num_neurons,
            size_t num_buffers);
  bool start();
  ~ChannelUpdater();
private:
  std::vector<ChannelSimulator<MType>*> simulators_;
  std::thread master_thread_;
  std::vector<std::thread> worker_threads_;
  typename NeuronPublisher::Subscription* neuron_subscription_;
  size_t num_buffers_;
  const ncs::spec::SimulationParameters* simulation_parameters_;
};

#include "Channel.hpp"
