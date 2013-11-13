#pragma once
#include <ncs/sim/DataBuffer.h>

template<ncs::sim::DeviceType::Type MType>
class NeuronBuffer : public ncs::sim::DataBuffer {
public:
  NeuronBuffer();
  bool init(size_t num_neurons);
  float* getVoltage();
  float* getCalcium();
  int* getSpikeShapeState();
  ~NeuronBuffer();
private:
  float* voltage_;
  float* calcium_;
  int* spike_shape_state_;
};

#include "NeuronBuffer.hpp"
