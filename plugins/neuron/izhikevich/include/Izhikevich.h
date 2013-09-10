#pragma once
#include <ncs/spec/Generator.h>

struct Instantiator {
  ncs::spec::Generator* a;
  ncs::spec::Generator* b;
  ncs::spec::Generator* c;
  ncs::spec::Generator* d;
  ncs::spec::Generator* u;
  ncs::spec::Generator* v;
};

class IzhikevichSimulator : public ncs::sim::NeuronSimulator {
public:
private:
};
