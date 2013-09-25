#pragma once

#include <list>
#include <queue>

#include <ncs/sim/InputSimulator.h>
#include <ncs/spec/Generator.h>

struct Instantiator {
  ncs::spec::Generator* amplitude;
};

enum class InputType {
  Voltage,
  Current
};

template<ncs::sim::DeviceType::Type MType, InputType IType>
class RectangularSimulator : public ncs::sim::InputSimulator<MType> {
public:
  virtual bool addInputs(const std::vector<ncs::sim::Input*>& inputs,
                         void* instantiator,
                         float start_time,
                         float end_time);
  virtual bool initialize();
private:
  struct Batch {
    float* amplitude;
    unsigned int* device_neuron_id;
    float start_time;
    float end_time;
    ~Batch();
  };
  struct BatchStartsLater {
    bool operator()(const Batch* a, const Batch* b) const;
  };
  std::priority_queue<Batch*,
                      std::vector<Batch*>,
                      BatchStartsLater> future_batches_;
};

#include "Rectangular.hpp"
