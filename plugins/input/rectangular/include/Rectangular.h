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
  virtual bool 
    initialize(const ncs::spec::SimulationParameters* simulation_parameters);
  virtual bool update(ncs::sim::InputUpdateParameters* parameters);
private:
  virtual bool update_(ncs::sim::InputUpdateParameters* parameters);
  struct Batch {
    float* amplitude;
    unsigned int* device_neuron_id;
    size_t count;
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
  std::list<Batch*> active_batches_;
};

template<>
bool RectangularSimulator<ncs::sim::DeviceType::CPU, InputType::Voltage>::
update_(ncs::sim::InputUpdateParameters* parameters);

template<>
bool RectangularSimulator<ncs::sim::DeviceType::CUDA, InputType::Voltage>::
update_(ncs::sim::InputUpdateParameters* parameters);

template<>
bool RectangularSimulator<ncs::sim::DeviceType::CPU, InputType::Current>::
update_(ncs::sim::InputUpdateParameters* parameters);

template<>
bool RectangularSimulator<ncs::sim::DeviceType::CUDA, InputType::Current>::
update_(ncs::sim::InputUpdateParameters* parameters);

#include "Rectangular.hpp"
