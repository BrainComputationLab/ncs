#pragma once
#include <ncs/sim/NeuronSimulator.h>
#include <ncs/spec/Generator.h>

#include <vector>

struct Instantiator {
  ncs::spec::Generator* a;
  ncs::spec::Generator* b;
  ncs::spec::Generator* c;
  ncs::spec::Generator* d;
  ncs::spec::Generator* u;
  ncs::spec::Generator* v;
  ncs::spec::Generator* threshold;
};

template<ncs::sim::DeviceType::Type MType>
class IzhikevichSimulator : public ncs::sim::NeuronSimulator<MType> {
public:
  IzhikevichSimulator();
  virtual bool addNeuron(ncs::sim::Neuron* neuron);
  virtual bool initialize(const ncs::spec::SimulationParameters* parameters);
  virtual bool initializeVoltages(float* plugin_voltages);
  virtual bool update(ncs::sim::NeuronUpdateParameters* parameters);
  virtual ~IzhikevichSimulator();
private:
  struct Buffers {
    std::vector<float> a;
    std::vector<float> b;
    std::vector<float> c;
    std::vector<float> d;
    std::vector<float> u;
    std::vector<float> v;
    std::vector<float> threshold;
  };
  Buffers* buffers_;
  float* a_;
  float* b_;
  float* c_;
  float* d_;
  float* u_;
  float* v_;
  float* threshold_;
  unsigned int num_neurons_;
  float step_dt_;
};

template<>
bool IzhikevichSimulator<ncs::sim::DeviceType::CPU>::
update(ncs::sim::NeuronUpdateParameters* parameters);

template<>
bool IzhikevichSimulator<ncs::sim::DeviceType::CUDA>::
update(ncs::sim::NeuronUpdateParameters* parameters);


#include "Izhikevich.hpp"
