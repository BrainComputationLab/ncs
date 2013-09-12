#include <ncs/sim/SynapseSimulator.h>
#include <ncs/spec/Generator.h>

struct Instantiator {
  ncs::spec::Generator* delay;
  ncs::spec::Generator* current;
};

template<ncs::sim::DeviceType::Type MType>
class FlatSimulator : public ncs::sim::SynapseSimulator<MType> {
public:
  FlatSimulator();
  virtual bool addSynapse(ncs::sim::Synapse* synapse);
  virtual bool initialize();
private:
  std::vector<float> cpu_current_;
  float* device_current_;
};

#include "Flat.hpp"
