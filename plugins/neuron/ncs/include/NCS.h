#pragma once
#include <ncs/sim/DataBuffer.h>
#include <ncs/sim/NeuronSimulator.h>
#include <ncs/spec/Generator.h>

enum ChannelType {
  VoltageGated = 0,
  CalciumGated = 1
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
};

struct VoltageGatedInstantiator : public ChannelInstantiator {
  ncs::spec::Generator* conductance;
  std::vector<VoltageGatedInstantiator*> particles;
};

struct NeuronInstantiator {
  ncs::spec::Generator* threshold;
  ncs::spec::Generator* resting_potential;
  ncs::spec::Generator* calcium;
  ncs::spec::Generator* calcium_spike_increment;
  ncs::spec::Generator* tau_calcium;
  ncs::spec::Generator* leak_reversal_potential;
  ncs::spec::Generator* leak_conductance;
  ncs::spec::Generator* tau_membrane;
  ncs::spec::Generator* r_membrane;
  std::vector<ChannelInstantiator*> channels;
};

template<ncs::sim::DeviceType::Type MType>
class ChannelCurrentBuffer : public ncs::sim::DataBuffer {
public:
  ChannelCurrentBuffer();
  bool init(size_t num_neurons);
  float* getCurrent();
  ~ChannelCurrentBuffer();
private:
  float* current_;
};

template<ncs::sim::DeviceType::Type MType>
class ChannelSimulator {
public:
  ChannelSimulator();
  bool addChannel(void* instantiator, ncs::sim::Neuron* neuron);
  bool initialize();
  virtual ~ChannelSimulator();
protected:
  virtual bool init_() = 0;
  std::vector<void*> instantiators_;
  std::vector<ncs::sim::Neuron*> neurons_;
  unsigned int* neuron_plugin_ids_;
  size_t num_channels_;
private:
};

template<ncs::sim::DeviceType::Type MType>
class VoltageGatedChannelSimulator : public ChannelSimulator<MType> {
public:
  VoltageGatedChannelSimulator();
  virtual ~VoltageGatedChannelSimulator();
protected:
  virtual bool init_();
private:
  struct ParticleConstants {
    ParticleConstants();
    float* a;
    float* b;
    float* c;
    float* d;
    float* f;
    float* h;
  };
  ParticleConstants alpha_;
  ParticleConstants beta_;
  unsigned int* particle_indices_;
  float* particle_products_;
};

template<ncs::sim::DeviceType::Type MType>
class ChannelUpdater 
  : public ncs::sim::SpecificPublisher<ChannelCurrentBuffer<MType>> {
public:
private:
};

#include "NCS.hpp"
