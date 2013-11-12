#pragma once
#include <thread>

#include <ncs/sim/DataBuffer.h>
#include <ncs/sim/NeuronSimulator.h>
#include <ncs/spec/Generator.h>
#include <ncs/spec/SimulationParameters.h>

#include "Channel.h"

enum ChannelType {
  VoltageGated = 0
};

struct ChannelInstantiator {
  ChannelType type;
};

struct ParticleConstantsInstantiator {
  ncs::spec::Generator* a;
  ncs::spec::Generator* b;
  ncs::spec::Generator* c;
  ncs::spec::Generator* d;
  ncs::spec::Generator* f;
  ncs::spec::Generator* h;
};

struct VoltageParticleInstantiator {
  ParticleConstantsInstantiator* alpha;
  ParticleConstantsInstantiator* beta;
  ncs::spec::Generator* power;
  ncs::spec::Generator* x_initial;
};

struct VoltageGatedInstantiator : public ChannelInstantiator {
  ncs::spec::Generator* conductance;
  ncs::spec::Generator* reversal_potential;
  std::vector<VoltageParticleInstantiator*> particles;
};

struct NeuronInstantiator {
  ncs::spec::Generator* threshold;
  ncs::spec::Generator* resting_potential;
  ncs::spec::Generator* capacitance;
  std::vector<ChannelInstantiator*> channels;
};

template<ncs::sim::DeviceType::Type MType>
struct ParticleConstants {
  ParticleConstants();
  bool init(size_t num_constants);
  template<ncs::sim::DeviceType::Type SType>
  bool copyFrom(ParticleConstants<SType>* source, size_t num_constants);
  ~ParticleConstants();
  float* a;
  float* b;
  float* c;
  float* d;
  float* f;
  float* h;
};

template<ncs::sim::DeviceType::Type MType>
class VoltageGatedChannelSimulator : public ChannelSimulator<MType> {
public:
  VoltageGatedChannelSimulator();
  virtual bool update(ChannelUpdateParameters* parameters);
  virtual ~VoltageGatedChannelSimulator();
protected:
  virtual bool init_();
private:
  size_t num_particles_;
  ParticleConstants<MType> alpha_;
  ParticleConstants<MType> beta_;
  unsigned int* particle_indices_;
  unsigned int* neuron_id_by_particle_;

  // num_particles_ in size
  float* x_;
  float* power_;

  // num_channels_ in size
  float* particle_products_;
  float* conductance_;
  float* reversal_potential_;
};

template<>
bool VoltageGatedChannelSimulator<ncs::sim::DeviceType::CPU>::
update(ChannelUpdateParameters* parameters);

template<>
bool VoltageGatedChannelSimulator<ncs::sim::DeviceType::CUDA>::
update(ChannelUpdateParameters* parameters);

template<ncs::sim::DeviceType::Type MType>
class HHSimulator 
  : public ncs::sim::NeuronSimulator<MType>,
    public ncs::sim::SpecificPublisher<NeuronBuffer<MType>> {
public:
  HHSimulator();
  virtual bool addNeuron(ncs::sim::Neuron* neuron);
  virtual bool initialize(const ncs::spec::SimulationParameters* parameters);
  virtual bool initializeVoltages(float* plugin_voltages);
  virtual bool update(ncs::sim::NeuronUpdateParameters* parameters);
  virtual ~HHSimulator();
private:
  std::vector<ChannelSimulator<MType>*> channel_simulators_;
  std::vector<ncs::sim::Neuron*> neurons_;
  size_t num_neurons_;
  ChannelUpdater<MType>* channel_updater_;
  typename ChannelUpdater<MType>::Subscription* channel_current_subscription_;
  typename  HHSimulator<MType>::Subscription* state_subscription_;
  const ncs::spec::SimulationParameters* simulation_parameters_;

  float* threshold_;
  float* resting_potential_;
  float* capacitance_;
};

template<>
bool HHSimulator<ncs::sim::DeviceType::CPU>::
update(ncs::sim::NeuronUpdateParameters* parameters);

template<>
bool HHSimulator<ncs::sim::DeviceType::CUDA>::
update(ncs::sim::NeuronUpdateParameters* parameters);

#include "HH.hpp"
