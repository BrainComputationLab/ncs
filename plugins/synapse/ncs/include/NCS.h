#include <ncs/sim/SynapseGenerator.h>
#include <ncs/spec/Generator.h>

struct Instantiator {
  ncs::spec::Generator* utilization;
  ncs::spec::Generator* redistribution;
  ncs::spec::Generator* last_fire_time;
  ncs::spec::Generator* max_conductance;
  ncs::spec::Generator* delay;
  ncs::spec::Generator* reversal_potential;
  ncs::spec::Generator* tau_facilitation;
};
