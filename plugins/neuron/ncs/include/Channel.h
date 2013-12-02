#pragma once
#include <thread>

#include <ncs/sim/DataBuffer.h>
#include <ncs/spec/Generator.h>
#include <ncs/spec/SimulationParameters.h>

#include "NeuronBuffer.h"

struct Channel {
  enum Type {
    VoltageGatedIon = 0,
    CalciumDependent = 1
  };
  Type type;
};

struct VoltageGatedIonChannel : public Channel {
  VoltageGatedIonChannel();
  ncs::spec::Generator* v_half;
  ncs::spec::Generator* r;
  ncs::spec::Generator* activation_slope;
  ncs::spec::Generator* deactivation_slope;
  ncs::spec::Generator* equilibrium_slope;
  ncs::spec::Generator* conductance;
  ncs::spec::Generator* reversal_potential;
  ncs::spec::Generator* m_initial;
  static VoltageGatedIonChannel* 
    instantiate(ncs::spec::ModelParameters* parameters);
};

struct CalciumDependentChannel : public Channel {
  CalciumDependentChannel();
  ncs::spec::Generator* m_initial;
  ncs::spec::Generator* reversal_potential;
  ncs::spec::Generator* conductance;
  ncs::spec::Generator* backwards_rate;
  ncs::spec::Generator* forward_scale;
  ncs::spec::Generator* forward_exponent;
  ncs::spec::Generator* tau_scale;
  ncs::spec::Generator* m_power;
  static CalciumDependentChannel*
    instantiate(ncs::spec::ModelParameters* parameters);
};

template<ncs::sim::DeviceType::Type MType>
class ChannelCurrentBuffer : public ncs::sim::DataBuffer {
public:
  ChannelCurrentBuffer();
  bool init(size_t num_neurons);
  void clear();
  float* getCurrent();
  ~ChannelCurrentBuffer();
private:
  float* current_;
  size_t num_neurons_;
};

struct ChannelUpdateParameters {
  const float* voltage;
  const float* calcium;
  float* current;
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
class VoltageGatedIonBuffer : public ncs::sim::DataBuffer {
public:
  VoltageGatedIonBuffer();
  bool init(size_t num_channels);
  float* getM();
  ~VoltageGatedIonBuffer();
private:
  float* m_;
};

template<ncs::sim::DeviceType::Type MType>
class VoltageGatedIonSimulator 
  : public ChannelSimulator<MType>,
    public ncs::sim::SpecificPublisher<VoltageGatedIonBuffer<MType>> {
public:
  VoltageGatedIonSimulator();
  virtual bool update(ChannelUpdateParameters* parameters);
  virtual ~VoltageGatedIonSimulator();
private:
  virtual bool init_();
  float* v_half_;
  float* tau_scale_factor_;
  float* activation_scale_;
  float* deactivation_scale_;
  float* equilibrium_scale_;
  float* conductance_;
  float* reversal_potential_;
  typedef ncs::sim::SpecificPublisher<VoltageGatedIonBuffer<MType>> Publisher;
  typename Publisher::Subscription* state_subscription_;
};

template<>
bool VoltageGatedIonSimulator<ncs::sim::DeviceType::CPU>::
update(ChannelUpdateParameters* parameters);

#ifdef NCS_CUDA
template<>
bool VoltageGatedIonSimulator<ncs::sim::DeviceType::CUDA>::
update(ChannelUpdateParameters* parameters);
#endif // NCS_CUDA

template<ncs::sim::DeviceType::Type MType>
class CalciumDependentBuffer : public ncs::sim::DataBuffer {
public:
  CalciumDependentBuffer();
  bool init(size_t num_channels);
  float* getM();
  ~CalciumDependentBuffer();
private:
  float* m_;
};

template<ncs::sim::DeviceType::Type MType>
class CalciumDependentSimulator 
  : public ChannelSimulator<MType>,
    public ncs::sim::SpecificPublisher<CalciumDependentBuffer<MType>> {
public:
  CalciumDependentSimulator();
  virtual bool update(ChannelUpdateParameters* parameters);
  virtual ~CalciumDependentSimulator();
private:
  virtual bool init_();
  float* reversal_potential_;
  float* conductance_;
  float* backwards_rate_;
  float* forward_scale_;
  float* forward_exponent_;
  float* tau_scale_;
  float* m_power_;
  typedef ncs::sim::SpecificPublisher<CalciumDependentBuffer<MType>> Publisher;
  typename Publisher::Subscription* state_subscription_;
};

template<>
bool CalciumDependentSimulator<ncs::sim::DeviceType::CPU>::
update(ChannelUpdateParameters* parameters);

#ifdef NCS_CUDA
template<>
bool CalciumDependentSimulator<ncs::sim::DeviceType::CUDA>::
update(ChannelUpdateParameters* parameters);
#endif // NCS_CUDA

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
#include "VoltageGatedIonChannel.hpp"
#include "CalciumDependentChannel.hpp"
