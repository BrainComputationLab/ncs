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
};

template<ncs::sim::DeviceType::Type MType>
class IzhikevichSimulator : public ncs::sim::NeuronSimulator<MType> {
public:
  IzhikevichSimulator();
  virtual bool addNeuron(ncs::sim::Neuron* neuron);
  virtual bool initialize();
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
  };
  Buffers* buffers_;
  float* a_;
  float* b_;
  float* c_;
  float* d_;
  float* u_;
  float* v_;
  unsigned int num_neurons_;
};

#include "Izhikevich.hpp"
