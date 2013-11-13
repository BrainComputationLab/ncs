#pragma once
#include <ncs/sim/DataBuffer.h>
#include <ncs/sim/NeuronSimulator.h>
#include <ncs/spec/ModelParameters.h>

#include "Channel.h"

struct SpikeShape {
  std::vector<float> voltages;
};

struct NCSNeuron {
  ncs::spec::Generator* threshold;
  ncs::spec::Generator* resting_potential;
  ncs::spec::Generator* calcium;
  ncs::spec::Generator* calcium_spike_increment;
  ncs::spec::Generator* tau_calcium;
  ncs::spec::Generator* leak_reversal_potential;
  ncs::spec::Generator* tau_membrane;
  ncs::spec::Generator* r_membrane;
  SpikeShape* spike_shape;
  std::vector<Channel*> channels;
  NCSNeuron();
  ~NCSNeuron();
  static void* instantiate(ncs::spec::ModelParameters* parameters);
};

template<ncs::sim::DeviceType::Type MType>
class NCSSimulator 
  : public ncs::sim::NeuronSimulator<MType>,
    public ncs::sim::SpecificPublisher<NeuronBuffer<MType>> {
public:
  NCSSimulator();
  virtual bool addNeuron(ncs::sim::Neuron* neuron);
  virtual bool initialize(const ncs::spec::SimulationParameters* parameters);
  virtual bool initializeVoltages(float* plugin_voltages);
  virtual bool update(ncs::sim::NeuronUpdateParameters* parameters);
  virtual ~NCSSimulator();
private:
  std::vector<ChannelSimulator<MType>*> channel_simulators_;
  std::vector<ncs::sim::Neuron*> neurons_;
  size_t num_neurons_;
  ChannelUpdater<MType>* channel_updater_;
  typename ChannelUpdater<MType>::Subscription* channel_current_subscription_;
  typename NCSSimulator<MType>::Subscription* state_subscription_;
  const ncs::spec::SimulationParameters* simulation_parameters_;

  float* threshold_;
  float* resting_potential_;
  float* calcium_;
  float* calcium_spike_increment_;
  float* tau_calcium_;
  float* leak_reversal_potential_;
  float* tau_membrane_;
  float* r_membrane_;

  float* spike_shape_data_;
  unsigned int* spike_shape_length_;
  float** spike_shape_;

  float* voltage_persistence_;
  float* dt_over_capacitance_;
  float* calcium_persistence_;
};

template<>
bool NCSSimulator<ncs::sim::DeviceType::CPU>::
update(ncs::sim::NeuronUpdateParameters* parameters);
#include "NCS.hpp"
