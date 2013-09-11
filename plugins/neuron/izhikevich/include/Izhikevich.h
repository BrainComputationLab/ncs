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

template<ncs::sim::DeviceType::Type MemoryType>
class IzhikevichSimulator : public ncs::sim::NeuronSimulator<MemoryType> {
public:
  virtual bool addNeuron(ncs::sim::Neuron* neuron);
  virtual bool initialize();
private:
  std::vector<float> a_;
  std::vector<float> b_;
  std::vector<float> c_;
  std::vector<float> d_;
  std::vector<float> u_;
  std::vector<float> v_;
};

#include "Izhikevich.hpp"
