#pragma once
#include <ncs/sim/NeuronSimulator.h>
#include <ncs/spec/Generator.h>

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
  virtual bool addNeuron(void* instantiator, unsigned int seed);
  virtual bool initialize();
private:
};

#include "Izhikevich.hpp"
